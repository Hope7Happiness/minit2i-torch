#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mini_t2i.train import build_model, build_text_encoder, encode_text, load_checkpoint_if_available
from mini_t2i.config import TrainConfig
from mini_t2i.diffusion import euler_sample
from mini_t2i.dpg_data import load_dpg_prompts as load_dpg_prompt_pairs
from mini_t2i import settings
from mini_t2i.utils import amp_dtype, rank, read_token, seed_all, world_size


DEFAULT_GENEVAL_METADATA = str(settings.GENEVAL_METADATA)
DEFAULT_DPG_DATA = str(settings.DPG_BENCH_DATA)


@dataclass(frozen=True)
class PromptItem:
    index: int
    item_id: str
    prompt: str
    metadata: dict | None = None


def init_dist() -> torch.device:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def load_geneval_prompts(path: str | os.PathLike[str]) -> list[PromptItem]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(PromptItem(i, f"{i:05d}", row["prompt"], row))
    return items


def load_dpg_prompts(source: str | os.PathLike[str]) -> list[PromptItem]:
    return [PromptItem(i, item_id, prompt) for i, (item_id, prompt) in enumerate(load_dpg_prompt_pairs(source))]


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = (images.detach().float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    arrays = images.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(array) for array in arrays]


def make_grid(images: list[Image.Image], image_size: int) -> Image.Image:
    if len(images) == 1:
        return images[0]
    if len(images) != 4:
        raise ValueError("DPG-Bench grid output expects 1 or 4 samples per prompt.")
    canvas = Image.new("RGB", (image_size * 2, image_size * 2))
    for image, xy in zip(images, [(0, 0), (image_size, 0), (0, image_size), (image_size, image_size)]):
        canvas.paste(image, xy)
    return canvas


def write_geneval(item: PromptItem, images: list[Image.Image], outdir: Path, overwrite: bool) -> None:
    prompt_dir = outdir / item.item_id
    sample_dir = prompt_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = prompt_dir / "metadata.jsonl"
    if overwrite or not metadata_path.exists():
        metadata_path.write_text(json.dumps(item.metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    for i, image in enumerate(images):
        path = sample_dir / f"{i:05d}.png"
        if overwrite or not path.exists():
            image.save(path)


def write_dpg(item: PromptItem, images: list[Image.Image], outdir: Path, image_size: int, overwrite: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{item.item_id}.png"
    if overwrite or not path.exists():
        make_grid(images, image_size).save(path)


def output_complete(benchmark: str, item: PromptItem, outdir: Path, samples_per_prompt: int) -> bool:
    if benchmark == "geneval":
        sample_dir = outdir / item.item_id / "samples"
        metadata_path = outdir / item.item_id / "metadata.jsonl"
        return metadata_path.exists() and all((sample_dir / f"{i:05d}.png").exists() for i in range(samples_per_prompt))
    return (outdir / f"{item.item_id}.png").exists()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name]


@torch.no_grad()
def generate_for_prompt(
    model,
    text_encoder,
    tokenizer,
    cfg: TrainConfig,
    item: PromptItem,
    device: torch.device,
    dtype: torch.dtype,
    samples_per_prompt: int,
    batch_size: int,
    seed: int,
) -> list[Image.Image]:
    pil_images: list[Image.Image] = []
    made = 0
    while made < samples_per_prompt:
        bsz = min(batch_size, samples_per_prompt - made)
        tok = tokenizer(
            [item.prompt] * bsz,
            max_length=cfg.prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        text = encode_text(text_encoder, input_ids, mask, dtype)
        torch.manual_seed(seed + item.index * 1000 + made + rank() * 1_000_000)
        images = euler_sample(
            model,
            text,
            mask,
            cfg.image_size,
            steps=cfg.n_T,
            cfg_scale=cfg.cfg_scale,
            noise_scale=cfg.noise_scale,
        )
        pil_images.extend(tensor_to_pil(images))
        made += bsz
    return pil_images


@torch.no_grad()
def generate_for_prompt_diffusers(
    pipe,
    model_type: str,
    repo_id_or_path: str,
    item: PromptItem,
    device: torch.device,
    samples_per_prompt: int,
    batch_size: int,
    seed: int,
    steps: int,
    cfg_scale: float,
) -> list[Image.Image]:
    pil_images: list[Image.Image] = []
    made = 0
    while made < samples_per_prompt:
        bsz = min(batch_size, samples_per_prompt - made)
        generator = torch.Generator(device=device).manual_seed(seed + item.index * 1000 + made + rank() * 1_000_000)
        out = pipe(
            [item.prompt] * bsz,
            model_type=model_type,
            repo_id_or_path=repo_id_or_path,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            generator=generator,
            progress=False,
        )
        pil_images.extend(out.images)
        made += bsz
    return pil_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GenEval or DPG-Bench images from a mini_t2i checkpoint.")
    parser.add_argument("--benchmark", choices=["geneval", "dpg"], required=True)
    parser.add_argument("--config-file", default="configs/finetune.yml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-target", choices=["model", "ema"], default="ema")
    parser.add_argument("--hf-model-id", default=None)
    parser.add_argument("--model-type", default="b16")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--geneval-metadata", default=DEFAULT_GENEVAL_METADATA)
    parser.add_argument("--dpg-data", default=DEFAULT_DPG_DATA)
    parser.add_argument("--dpg-csv", dest="dpg_data", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--samples-per-prompt", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--attention-impl", choices=["einsum", "sdpa"], default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.config_file:
        cfg.update_from_yaml(args.config_file)
    cfg.resume_from = args.checkpoint
    cfg.auto_resume = False
    if args.cfg_scale is not None:
        cfg.cfg_scale = args.cfg_scale
    if args.steps is not None:
        cfg.n_T = args.steps
    if args.amp_dtype is not None:
        cfg.amp_dtype = args.amp_dtype
    if args.attention_impl is not None:
        cfg.attention_impl = args.attention_impl
    cfg.resolve()

    samples_per_prompt = args.samples_per_prompt
    if samples_per_prompt is None:
        samples_per_prompt = 4
    if args.benchmark == "dpg" and samples_per_prompt not in (1, 4):
        raise ValueError("DPG-Bench supports --samples-per-prompt 1 or 4.")

    device = init_dist()
    seed_all(args.seed)
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompts = load_geneval_prompts(args.geneval_metadata) if args.benchmark == "geneval" else load_dpg_prompts(args.dpg_data)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    local_prompts = prompts[rank() :: world_size()]

    outdir = Path(args.outdir)
    if not args.overwrite:
        incomplete_prompts = [
            item for item in local_prompts if not output_complete(args.benchmark, item, outdir, samples_per_prompt)
        ]
    else:
        incomplete_prompts = local_prompts

    print(
        f"[generate] rank={rank()}/{world_size()} benchmark={args.benchmark} prompts={len(local_prompts)}/{len(prompts)} "
        f"samples_per_prompt={samples_per_prompt} steps={cfg.n_T} cfg_scale={cfg.cfg_scale} outdir={outdir}",
        flush=True,
    )
    if not incomplete_prompts:
        print(f"[generate] rank={rank()} all outputs complete; skipping model load", flush=True)
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
            dist.destroy_process_group()
        return

    if args.hf_model_id:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            args.hf_model_id,
            custom_pipeline=args.hf_model_id,
            torch_dtype=dtype,
            cache_dir=args.cache_dir,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        for n, item in enumerate(incomplete_prompts, start=1):
            images = generate_for_prompt_diffusers(
                pipe,
                args.model_type,
                args.hf_model_id,
                item,
                device,
                samples_per_prompt,
                args.batch_size,
                args.seed,
                cfg.n_T,
                cfg.cfg_scale,
            )
            if args.benchmark == "geneval":
                write_geneval(item, images, outdir, args.overwrite)
            else:
                write_dpg(item, images, outdir, cfg.image_size, args.overwrite)
            print(f"[generate] rank={rank()} {n}/{len(incomplete_prompts)} saved {item.item_id}", flush=True)
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
            dist.destroy_process_group()
        return

    if not args.checkpoint:
        raise ValueError("Either --checkpoint or --hf-model-id must be provided.")

    text_encoder = build_text_encoder(cfg, device)
    model = build_model(cfg).to(device)
    ema_model = build_model(cfg).to(device).eval().requires_grad_(False)
    load_checkpoint_if_available(cfg, model, ema_model, device)
    sample_model = ema_model if args.checkpoint_target == "ema" else model
    sample_model.eval().to(dtype=dtype)

    token = read_token(cfg.hf_token_file)
    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name, token=token, model_max_length=cfg.prompt_length)

    for n, item in enumerate(incomplete_prompts, start=1):
        images = generate_for_prompt(
            sample_model,
            text_encoder,
            tokenizer,
            cfg,
            item,
            device,
            dtype,
            samples_per_prompt,
            args.batch_size,
            args.seed,
        )
        if args.benchmark == "geneval":
            write_geneval(item, images, outdir, args.overwrite)
        else:
            write_dpg(item, images, outdir, cfg.image_size, args.overwrite)
        print(f"[generate] rank={rank()} {n}/{len(incomplete_prompts)} saved {item.item_id}", flush=True)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
