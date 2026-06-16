import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from train_lora import find_minit2i_transformer, load_hf_transformer

transformers_logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images with a MiniT2I PEFT LoRA adapter.")
    parser.add_argument("--adapter_dir", required=True, help="Directory containing adapter_model.safetensors.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=None,
        help=(
            "Hub id or local Diffusers-format MiniT2I model. Defaults to adapter metadata "
            "or MiniT2I/MiniT2I."
        ),
    )
    parser.add_argument("--model_type", default=None, help="MiniT2I model type. Defaults to adapter metadata or b16.")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--outdir", default="lora_outputs")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def load_json_if_exists(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_adapter_metadata(adapter_dir: Path):
    metadata = load_json_if_exists(adapter_dir / "training_metadata.json")
    training_args = load_json_if_exists(adapter_dir / "training_args.json")
    adapter_config = load_json_if_exists(adapter_dir / "adapter_config.json")
    return {
        "training_metadata": metadata,
        "training_args": training_args,
        "adapter_config": adapter_config,
    }


def is_local_path(value):
    if value is None:
        return False
    return value.startswith((".", "/", "~")) or Path(value).exists()


def resolve_base_model(args, metadata):
    training_metadata = metadata["training_metadata"]
    training_args = metadata["training_args"]
    model_source = training_metadata.get("model_source") or training_args.get("model_source") or {}

    model_type = args.model_type or model_source.get("model_type") or training_args.get("model_type") or "b16"
    pretrained = args.pretrained_model_name_or_path
    if pretrained is None and model_source.get("model_source") == "hf":
        pretrained = model_source.get("pretrained_model_name_or_path")
    if pretrained is None:
        pretrained = training_args.get("base_model")
    if pretrained is None:
        pretrained = "MiniT2I/MiniT2I"

    if model_source.get("model_source") == "local_ckpt":
        if args.pretrained_model_name_or_path is None:
            raise ValueError(
                "This adapter metadata says it was trained from an old local .pt checkpoint. "
                "The MiniT2I sampler does not load .pt base checkpoints. Export that base model "
                "to Diffusers format and pass it with --pretrained_model_name_or_path, or sample "
                "an adapter trained from the MiniT2I Hub base (`MiniT2I/MiniT2I`)."
            )
        if not is_local_path(args.pretrained_model_name_or_path):
            raise ValueError(
                "This adapter metadata says it was trained from an old local .pt checkpoint, "
                f"but --pretrained_model_name_or_path was set to Hub model '{args.pretrained_model_name_or_path}'. "
                "That is a base-model mismatch and can produce invalid samples. Export the matching "
                "local base checkpoint to Diffusers format and pass that local directory, or use an "
                "adapter trained from the MiniT2I Hub base (`MiniT2I/MiniT2I`)."
            )
    return pretrained, model_type


def tensor_to_pil(images):
    imgs = (images.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in imgs]


def load_model(args, metadata, dtype):
    from peft import PeftModel

    pretrained, model_type = resolve_base_model(args, metadata)
    transformer, cfg = load_hf_transformer(
        pretrained_model_name_or_path=pretrained,
        model_type=model_type,
        cache_dir=args.cache_dir,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
    )
    transformer.requires_grad_(False)
    transformer = PeftModel.from_pretrained(transformer, args.adapter_dir, is_trainable=False)
    transformer.eval()
    return transformer, cfg, pretrained, model_type


def encode_prompt(tokenizer, text_encoder, prompt, cfg, device, dtype):
    tokens = tokenizer(
        [prompt],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.prompt_length,
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return prompt_embeds.to(dtype=dtype), attention_mask.to(device)


@torch.no_grad()
def sample(args):
    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Expected PEFT adapter weights at {adapter_dir / 'adapter_model.safetensors'}")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    metadata = load_adapter_metadata(adapter_dir)
    transformer, cfg, pretrained, model_type = load_model(args, metadata, dtype)
    transformer.to(device=device, dtype=dtype)
    model = find_minit2i_transformer(transformer)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(cfg.llm, torch_dtype=torch.float32, cache_dir=args.cache_dir)
    text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    old_steps = model.model.cfg.n_T
    manifest = ["prompt_idx\timage_idx\tseed\tprompt\tfile"]
    try:
        model.model.cfg.n_T = args.steps
        for prompt_idx, prompt in enumerate(args.prompt):
            prompt_embeds, attention_mask = encode_prompt(tokenizer, text_encoder, prompt, cfg, device, dtype)
            for image_idx in range(args.num_images_per_prompt):
                seed = args.seed + prompt_idx * args.num_images_per_prompt + image_idx
                generator = torch.Generator(device=device).manual_seed(seed)
                images = model.sample(
                    prompt_embeds,
                    attention_mask.to(dtype=dtype),
                    cfg_scale=args.cfg_scale,
                    generator=generator,
                    progress=True,
                )
                image = tensor_to_pil(images)[0]
                filename = f"sample_prompt_{prompt_idx:02d}_seed_{seed:06d}.png"
                image.save(outdir / filename)
                manifest.append(f"{prompt_idx}\t{image_idx}\t{seed}\t{prompt}\t{filename}")
    finally:
        model.model.cfg.n_T = old_steps

    (outdir / "samples.tsv").write_text("\n".join(manifest) + "\n")
    run_info = {
        "adapter_dir": str(adapter_dir),
        "pretrained_model_name_or_path": pretrained,
        "model_type": model_type,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "dtype": args.dtype,
    }
    (outdir / "sample_config.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"Saved samples to {outdir}")


def main():
    sample(parse_args())


if __name__ == "__main__":
    main()
