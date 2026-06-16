import argparse
import importlib
import io
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DIFFUSERS_DIR = Path(__file__).resolve().parents[1] / "diffusers"
sys.path.insert(0, str(DIFFUSERS_DIR))

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from pipeline import MiniT2IFlowMatchScheduler, MiniT2IMMJiTModel

transformers_logging.set_verbosity_error()
logger = get_logger(__name__)


MODEL_ALIASES = {
    "b": "minit2i-b-16",
    "b16": "minit2i-b-16",
    "b-16": "minit2i-b-16",
    "base": "minit2i-b-16",
    "minit2i-b16": "minit2i-b-16",
    "minit2i-b-16": "minit2i-b-16",
    "minit2i-b/16": "minit2i-b-16",
    "l": "minit2i-l-16",
    "l16": "minit2i-l-16",
    "l-16": "minit2i-l-16",
    "large": "minit2i-l-16",
    "minit2i-l16": "minit2i-l-16",
    "minit2i-l-16": "minit2i-l-16",
    "minit2i-l/16": "minit2i-l-16",
}

DEFAULT_TARGET_MODULES = ("qkv", "attn_proj", "w1", "w2", "w3", "txt_embedder", "pooled_embedder")


def import_optional(module_name: str, install_name: Optional[str] = None):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = install_name or module_name
        raise ImportError(
            f"`{module_name}` is required for this code path. Install it with `pip install {package}` "
            "or choose a compatible fallback."
        ) from exc


def resolve_model_type(model_type: str) -> str:
    key = model_type.lower().replace("_", "-")
    if key not in MODEL_ALIASES:
        choices = ", ".join(sorted(set(MODEL_ALIASES)))
        raise ValueError(f"Unknown model_type={model_type!r}. Expected one of: {choices}")
    return MODEL_ALIASES[key]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def find_minit2i_transformer(module: nn.Module, _seen: Optional[set] = None) -> MiniT2IMMJiTModel:
    """Find the MiniT2IMMJiTModel inside optional wrappers such as PEFT/Accelerate."""

    module = unwrap_model(module)
    if isinstance(module, MiniT2IMMJiTModel):
        return module
    if _seen is None:
        _seen = set()
    if id(module) in _seen:
        raise TypeError(f"Could not locate MiniT2IMMJiTModel inside {type(module).__name__}")
    _seen.add(id(module))

    if hasattr(module, "get_base_model"):
        try:
            return find_minit2i_transformer(module.get_base_model(), _seen)
        except Exception:
            pass
    for attr in ("base_model", "model"):
        child = getattr(module, attr, None)
        if isinstance(child, nn.Module):
            try:
                return find_minit2i_transformer(child, _seen)
            except Exception:
                pass
    raise TypeError(f"Could not locate MiniT2IMMJiTModel inside {type(module).__name__}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MiniT2I LoRA adapter.")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--pretrained_model_name_or_path",
        default="MiniT2I/MiniT2I",
        help=(
            "Hub id or local Diffusers-format MiniT2I model. Defaults to MiniT2I/MiniT2I."
        ),
    )
    model_group.add_argument("--model_type", default="b16")
    model_group.add_argument("--revision", default=None)
    model_group.add_argument("--variant", default=None)
    model_group.add_argument("--cache_dir", default=None)

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset_name", default=None)
    data_group.add_argument("--train_data_dir", default=None)
    data_group.add_argument("--image_column", default="image")
    data_group.add_argument("--caption_column", default="text")
    data_group.add_argument("--caption_prefix", default="")
    data_group.add_argument("--max_caption_chars", type=int, default=0)
    data_group.add_argument("--max_train_samples", type=int, default=None)
    data_group.add_argument("--resolution", type=int, default=None)
    data_group.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=True)
    data_group.add_argument("--random_flip", action="store_true")
    data_group.add_argument("--dataloader_num_workers", type=int, default=2)
    data_group.add_argument("--dataloader_drop_last", action=argparse.BooleanOptionalAction, default=True)

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--output_dir", default="minit2i-lora-output")
    train_group.add_argument("--train_batch_size", type=int, default=1)
    train_group.add_argument("--num_train_epochs", type=int, default=1)
    train_group.add_argument("--max_train_steps", type=int, default=None)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train_group.add_argument("--learning_rate", type=float, default=1e-4)
    train_group.add_argument("--scale_lr", action="store_true")
    train_group.add_argument("--lr_scheduler", default="constant")
    train_group.add_argument("--lr_warmup_steps", type=int, default=0)
    train_group.add_argument("--adam_beta1", type=float, default=0.9)
    train_group.add_argument("--adam_beta2", type=float, default=0.999)
    train_group.add_argument("--adam_weight_decay", type=float, default=0.0)
    train_group.add_argument("--adam_epsilon", type=float, default=1e-8)
    train_group.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--data_seed", type=int, default=None, help="Seed for training DataLoader shuffle. Defaults to --seed.")
    train_group.add_argument("--train_rng_seed", type=int, default=None, help="Seed for training timesteps/noise. Defaults to --seed.")
    train_group.add_argument("--max_grad_norm", type=float, default=1.0)

    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--rank", type=int, default=16)
    lora_group.add_argument("--lora_alpha", type=int, default=None)
    lora_group.add_argument("--lora_dropout", type=float, default=0.0)
    lora_group.add_argument("--lora_init_seed", type=int, default=None, help="Seed for LoRA A/B initialization. Defaults to --seed.")
    lora_group.add_argument("--target_modules", default=None)

    loss_group = parser.add_argument_group("MiniT2I loss")
    loss_group.add_argument("--loss_type", choices=["velocity", "x0"], default="velocity")
    loss_group.add_argument("--t_schedule", choices=["lognorm", "uniform"], default="lognorm")
    loss_group.add_argument("--t_lognorm_mu", type=float, default=-0.8)
    loss_group.add_argument("--t_lognorm_sigma", type=float, default=0.8)
    loss_group.add_argument("--scheduler_dir", default=None)

    val_group = parser.add_argument_group("Validation")
    val_group.add_argument("--validation_prompt", action="append", default=None)
    val_group.add_argument("--num_validation_images", type=int, default=1)
    val_group.add_argument("--validation_steps", type=int, default=50)
    val_group.add_argument("--validation_guidance_scale", type=float, default=6.0)
    val_group.add_argument("--validation_steps_interval", type=int, default=500)
    val_group.add_argument("--validation_seed", type=int, default=0)

    log_group = parser.add_argument_group("Logging and checkpoints")
    log_group.add_argument("--report_to", default="none", help="Tracker for accelerate: none, tensorboard, wandb, etc.")
    log_group.add_argument("--logging_dir", default="logs")
    log_group.add_argument("--checkpointing_steps", type=int, default=500)
    log_group.add_argument("--checkpoints_total_limit", type=int, default=None)
    log_group.add_argument("--resume_from_checkpoint", default=None)

    val_group.add_argument("--validation_at_start", action="store_true")

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Pass --dataset_name or --train_data_dir.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive.")
    if args.lora_alpha is None:
        args.lora_alpha = args.rank
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.train_rng_seed is None:
        args.train_rng_seed = args.seed
    if args.lora_init_seed is None:
        args.lora_init_seed = args.seed


def resolve_model_source(args) -> Dict[str, Any]:
    return {
        "model_source": "hf",
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "model_type": args.model_type,
        "revision": args.revision,
        "variant": args.variant,
    }


def load_hf_transformer(
    pretrained_model_name_or_path: str,
    model_type: str,
    cache_dir: Optional[str],
    revision: Optional[str],
    variant: Optional[str],
    torch_dtype: Optional[torch.dtype],
):
    model_dir = resolve_model_type(model_type)
    root = Path(pretrained_model_name_or_path)
    if root.exists():
        if (root / "transformer").exists():
            transformer_dir = root / "transformer"
        else:
            transformer_dir = root / model_dir / "transformer"
    else:
        root = Path(
            snapshot_download(
                repo_id=pretrained_model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=[f"{model_dir}/transformer/*"],
            )
        )
        transformer_dir = root / model_dir / "transformer"
    transformer = MiniT2IMMJiTModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype, variant=variant)
    return transformer, transformer.mmjit_config


def load_or_create_scheduler(args) -> MiniT2IFlowMatchScheduler:
    if args.scheduler_dir:
        return MiniT2IFlowMatchScheduler.from_pretrained(args.scheduler_dir)
    return MiniT2IFlowMatchScheduler(
        train_t_schedule=args.t_schedule,
        t_lognorm_mu=args.t_lognorm_mu,
        t_lognorm_sigma=args.t_lognorm_sigma,
        num_inference_steps=args.validation_steps,
    )


def load_minit2i_model_components(args, weight_dtype: torch.dtype):
    model_source = resolve_model_source(args)
    transformer, cfg = load_hf_transformer(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        model_type=args.model_type,
        cache_dir=args.cache_dir,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    scheduler = load_or_create_scheduler(args)
    return transformer, cfg, scheduler, model_source


def load_text_components(cfg, args):
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(cfg.llm, torch_dtype=torch.float32, cache_dir=args.cache_dir)
    return tokenizer, text_encoder


def load_minit2i_components(args, weight_dtype: torch.dtype):
    transformer, cfg, scheduler, model_source = load_minit2i_model_components(args, weight_dtype)
    tokenizer, text_encoder = load_text_components(cfg, args)
    return transformer, cfg, scheduler, tokenizer, text_encoder, model_source


def parse_target_modules(text: Optional[str]) -> List[str]:
    if not text:
        return list(DEFAULT_TARGET_MODULES)
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_lora_target_module_names(model: nn.Module, target_modules: Sequence[str]) -> List[str]:
    """Resolve user-friendly target patterns to exact Linear module names for PEFT."""

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules)
    ]


def _peft_lora_weight(container) -> torch.Tensor:
    if isinstance(container, nn.ModuleDict):
        return container["default"].weight
    if isinstance(container, nn.Linear):
        return container.weight
    if hasattr(container, "weight"):
        return container.weight
    raise TypeError(f"Unsupported PEFT LoRA container type: {type(container).__name__}")


def initialize_peft_lora_weights(model: nn.Module, target_names: Sequence[str], rng_state: torch.Tensor):
    """Initialize PEFT LoRA weights with Kaiming A and zero B."""

    raw_modules = {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    }
    old_rng_state = torch.random.get_rng_state()
    torch.random.set_rng_state(rng_state)
    try:
        for target_name in target_names:
            matches = [module for name, module in raw_modules.items() if name.endswith(target_name)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one PEFT LoRA module ending with {target_name!r}, found {len(matches)}")
            module = matches[0]
            lora_a = _peft_lora_weight(module.lora_A)
            lora_b = _peft_lora_weight(module.lora_B)
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
            nn.init.zeros_(lora_b)
    finally:
        torch.random.set_rng_state(old_rng_state)


def add_lora_adapters(transformer: nn.Module, args) -> Dict[str, Any]:
    target_modules = parse_target_modules(args.target_modules)
    resolved_targets = resolve_lora_target_module_names(transformer, target_modules)
    if not resolved_targets:
        raise ValueError(f"No Linear modules matched target_modules={target_modules}")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("PEFT is required for MiniT2I LoRA training. Install it with `pip install peft`.") from exc

    lora_init_generator = torch.Generator()
    lora_init_generator.manual_seed(args.lora_init_seed)
    lora_init_rng_state = lora_init_generator.get_state()
    old_rng_state = torch.random.get_rng_state()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights=True,
        target_modules=resolved_targets,
    )
    try:
        if hasattr(transformer, "add_adapter"):
            transformer.add_adapter(lora_config)
            lora_model = transformer
        else:
            lora_model = get_peft_model(transformer, lora_config)
        initialize_peft_lora_weights(lora_model, resolved_targets, lora_init_rng_state)
    finally:
        torch.random.set_rng_state(old_rng_state)
    return {
        "backend": "peft",
        "init_order": "after_text_encoder",
        "init_seed": args.lora_init_seed,
        "target_modules": target_modules,
        "peft_target_modules": resolved_targets,
        "resolved_target_modules": resolved_targets,
        "lora_config": lora_config,
        "model": lora_model,
    }


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def cast_trainable_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    for param in trainable_parameters(model):
        param.data = param.data.to(dtype)


def parameter_summary(model: nn.Module) -> Dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return {"total_params": total, "trainable_params": trainable, "trainable_percent": pct}


def load_train_dataset(args):
    datasets = import_optional("datasets")
    from datasets import load_dataset

    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    elif args.train_data_dir:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir)
    else:
        raise ValueError("Pass --dataset_name or --train_data_dir.")
    if isinstance(dataset, dict):
        if "train" not in dataset:
            first_key = next(iter(dataset))
            return dataset[first_key]
        return dataset["train"]
    return dataset


def build_image_transform(args, image_size: int):
    resolution = args.resolution or image_size
    crop = transforms.CenterCrop(resolution) if args.center_crop else transforms.RandomCrop(resolution)
    transform_list = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        crop,
    ]
    if args.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_list)


def image_from_example(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value)
    return value.convert("RGB")


def preprocess_dataset(dataset, args, cfg):
    image_transform = build_image_transform(args, cfg.image_size)

    def preprocess(examples):
        images = []
        captions = []
        for image_value, caption_value in zip(examples[args.image_column], examples[args.caption_column]):
            image = ImageOps.exif_transpose(image_from_example(image_value)).convert("RGB")
            caption = caption_value
            if isinstance(caption, list):
                caption = random.choice(caption)
            caption = str(caption)
            if args.caption_prefix:
                caption = args.caption_prefix + caption
            if args.max_caption_chars and len(caption) > args.max_caption_chars:
                caption = caption[: args.max_caption_chars]
            images.append(image_transform(image))
            captions.append(caption)
        return {"pixel_values": images, "captions": captions}

    columns = dataset.column_names
    dataset = dataset.with_transform(preprocess)
    if args.max_train_samples is not None:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(dataset))))
    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    captions = [example["captions"] for example in examples]
    return {"pixel_values": pixel_values, "captions": captions}


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def encode_batch_prompts(tokenizer, text_encoder, captions, cfg, device, weight_dtype):
    tokens = tokenizer(
        captions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.prompt_length,
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return prompt_embeds.to(dtype=weight_dtype), attention_mask.to(device)


def compute_minit2i_loss(transformer, scheduler, images, prompt_embeds, attention_mask, args, weight_dtype, train_generator=None):
    batch_size = images.shape[0]
    t = scheduler.sample_train_timesteps(batch_size, device=images.device, dtype=weight_dtype, generator=train_generator)
    noise = torch.randn(
        images.shape,
        device=images.device,
        dtype=weight_dtype,
        generator=train_generator,
    ) * 2
    x_t = images * t[:, None, None, None] + noise * (1 - t[:, None, None, None])
    target = (images - x_t) / torch.clamp(1 - t[:, None, None, None], min=0.05)
    model = find_minit2i_transformer(transformer)

    if args.loss_type == "x0":
        pred_x0 = model.model.net(
            x_t,
            model.model.real_t_to_embed_t(t),
            prompt_embeds,
            attention_mask.to(dtype=weight_dtype),
        )
        loss = F.mse_loss(pred_x0.float(), images.float(), reduction="mean")
    else:
        pred = model.pred_velocity(
            x_t,
            t,
            prompt_embeds,
            attention_mask.to(dtype=weight_dtype),
        )
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
    logs = {
        "t_mean": float(t.detach().float().mean().cpu()),
        "t_min": float(t.detach().float().min().cpu()),
        "t_max": float(t.detach().float().max().cpu()),
    }
    return loss, logs


def tensor_to_pil(images):
    imgs = (images.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in imgs]


@torch.no_grad()
def run_validation(transformer, tokenizer, text_encoder, cfg, args, accelerator: Accelerator, global_step: int, weight_dtype):
    if not args.validation_prompt or not accelerator.is_main_process:
        return
    validation_dir = Path(args.output_dir) / "validation_images"
    validation_dir.mkdir(parents=True, exist_ok=True)
    model = find_minit2i_transformer(accelerator.unwrap_model(transformer))
    was_training = model.training
    old_steps = model.model.cfg.n_T
    model.eval()
    model.model.cfg.n_T = args.validation_steps
    try:
        for prompt_idx, prompt in enumerate(args.validation_prompt):
            for image_idx in range(args.num_validation_images):
                prompt_embeds, attention_mask = encode_batch_prompts(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    captions=[prompt],
                    cfg=cfg,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )
                seed = args.validation_seed + prompt_idx * args.num_validation_images + image_idx
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                images = model.sample(
                    prompt_embeds,
                    attention_mask.to(dtype=weight_dtype),
                    cfg_scale=args.validation_guidance_scale,
                    generator=generator,
                    progress=False,
                )
                pil = tensor_to_pil(images)[0]
                name = f"step_{global_step:06d}_prompt_{prompt_idx:02d}_seed_{seed:06d}.png"
                pil.save(validation_dir / name)
                accelerator.log({f"validation/prompt_{prompt_idx:02d}": prompt}, step=global_step)
    finally:
        model.model.cfg.n_T = old_steps
        model.train(was_training)


def prune_checkpoints(output_dir: Path, total_limit: Optional[int]):
    if not total_limit or total_limit <= 0:
        return
    checkpoints = sorted(path for path in output_dir.glob("checkpoint-*") if path.is_dir())
    excess = len(checkpoints) - total_limit
    for path in checkpoints[: max(excess, 0)]:
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()


def peft_state_dict(model: nn.Module):
    try:
        from peft.utils import get_peft_model_state_dict
    except ImportError as exc:
        raise ImportError("Cannot save PEFT LoRA weights because `peft` is not installed.") from exc
    return get_peft_model_state_dict(model)


def save_lora_adapter(transformer, args, accelerator: Accelerator, lora_info: Dict[str, Any], model_source, global_step=None, final=False):
    if not accelerator.is_main_process:
        return
    output_dir = Path(args.output_dir)
    save_dir = output_dir if final else output_dir / f"lora-{global_step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(transformer)
    metadata = {
        "format": "peft",
        "base_model": args.pretrained_model_name_or_path,
        "model_type": args.model_type,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": lora_info["target_modules"],
        "resolved_target_modules": lora_info.get("resolved_target_modules", []),
        "resolved_target_module_count": len(lora_info.get("resolved_target_modules", [])),
        "lora_init_seed": lora_info.get("init_seed"),
        "loss_type": args.loss_type,
        "data_seed": args.data_seed,
        "train_rng_seed": args.train_rng_seed,
        "model_source": model_source,
        "minit2i_train_lora_version": 3,
    }
    (save_dir / "training_args.json").write_text(json.dumps(metadata, indent=2) + "\n")

    safetensors = import_optional("safetensors.torch", "safetensors")
    state_dict = peft_state_dict(unwrapped)
    safetensors.save_file(
        {key: value.detach().cpu() for key, value in state_dict.items()},
        save_dir / "adapter_model.safetensors",
    )
    lora_config = lora_info.get("lora_config")
    if hasattr(lora_config, "save_pretrained"):
        lora_config.save_pretrained(save_dir)


def save_training_metadata(args, model_source, lora_info):
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": vars(args),
        "model_source": model_source,
        "adapter_format": "peft",
        "target_modules": lora_info["target_modules"],
        "resolved_target_modules": lora_info.get("resolved_target_modules", []),
        "resolved_target_module_count": len(lora_info.get("resolved_target_modules", [])),
        "lora_init_seed": lora_info.get("init_seed"),
        "data_seed": args.data_seed,
        "train_rng_seed": args.train_rng_seed,
    }
    (Path(args.output_dir) / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_lr_scheduler(args, optimizer, accelerator: Accelerator):
    try:
        from diffusers.optimization import get_scheduler
    except Exception:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max(args.max_train_steps or 1, 1) * accelerator.num_processes,
    )


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    output_dir = Path(args.output_dir)
    logging_dir = output_dir / args.logging_dir
    project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer, cfg, scheduler, tokenizer, text_encoder, model_source = load_minit2i_components(args, weight_dtype)
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    lora_info = add_lora_adapters(transformer, args)
    if "model" in lora_info:
        transformer = lora_info["model"]

    transformer.to(accelerator.device, dtype=weight_dtype)
    cast_trainable_parameters(transformer, torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32).eval()

    summary = parameter_summary(transformer)
    logger.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        summary["trainable_params"],
        summary["total_params"],
        summary["trainable_percent"],
        main_process_only=False,
    )

    train_dataset = preprocess_dataset(load_train_dataset(args), args, cfg)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.data_seed)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=args.dataloader_drop_last,
        generator=data_generator,
        worker_init_fn=seed_worker,
    )

    if args.max_train_steps is None:
        updates_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * updates_per_epoch
    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer = torch.optim.AdamW(
        trainable_parameters(transformer),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = build_lr_scheduler(args, optimizer, accelerator)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_training_metadata(args, model_source, lora_info)
    if args.report_to != "none":
        accelerator.init_trackers("minit2i-lora", config=vars(args))

    global_step = 0
    first_epoch = 0
    transformer.train()
    train_generator = torch.Generator(device=accelerator.device)
    train_generator.manual_seed(args.train_rng_seed)
    if args.validation_at_start:
        run_validation(transformer, tokenizer, text_encoder, cfg, args, accelerator, global_step, weight_dtype)
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                images = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                prompt_embeds, attention_mask = encode_batch_prompts(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    captions=batch["captions"],
                    cfg=cfg,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )
                loss, loss_logs = compute_minit2i_loss(
                    transformer=transformer,
                    scheduler=scheduler,
                    images=images,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    args=args,
                    weight_dtype=weight_dtype,
                    train_generator=train_generator,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters(transformer), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                logs = {
                    "train/loss": float(loss.detach().cpu()),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    **{f"train/{key}": value for key, value in loss_logs.items()},
                }
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(checkpoint_dir))
                    save_lora_adapter(transformer, args, accelerator, lora_info, model_source, global_step=global_step)
                    prune_checkpoints(output_dir, args.checkpoints_total_limit)

                if (
                    args.validation_prompt
                    and args.validation_steps_interval > 0
                    and global_step % args.validation_steps_interval == 0
                ):
                    run_validation(transformer, tokenizer, text_encoder, cfg, args, accelerator, global_step, weight_dtype)

                if global_step >= args.max_train_steps:
                    break
        else:
            first_epoch += 1
            if first_epoch >= args.num_train_epochs and args.max_train_steps is None:
                break

    accelerator.wait_for_everyone()
    save_lora_adapter(transformer, args, accelerator, lora_info, model_source, global_step=global_step, final=True)
    if args.validation_prompt:
        run_validation(transformer, tokenizer, text_encoder, cfg, args, accelerator, global_step, weight_dtype)
    accelerator.end_training()
    if accelerator.is_main_process:
        print(f"Saved LoRA adapter to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
