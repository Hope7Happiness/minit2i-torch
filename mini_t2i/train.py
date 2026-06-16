from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5EncoderModel

from mini_t2i.config import TrainConfig
from mini_t2i.data import check_training_data, make_loader
from mini_t2i.diffusion import training_loss, euler_sample
from mini_t2i.fid import InceptionFeatures, compute_fid_from_images, extract_features, fid_from_features, load_captions
from mini_t2i.model import MMJiTB32Text2
from mini_t2i.settings import WANDB_API_KEY_FILE
from mini_t2i.utils import (
    amp_dtype,
    atomic_torch_save,
    is_main,
    jax_warmup_constant_lr,
    rank,
    read_token,
    seed_all,
    update_ema,
    world_size,
)
from mini_t2i.visualize import save_grid


def init_dist() -> tuple[torch.device, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank


def build_model(cfg: TrainConfig) -> MMJiTB32Text2:
    return MMJiTB32Text2(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        t5_hidden_size=cfg.t5_hidden_size,
        depth_double=cfg.depth_double,
        text_preamble_depth=cfg.text_preamble_depth,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        mlp_ratio=cfg.mlp_ratio,
        pca_channels=cfg.pca_channels,
        final_layer_zero=cfg.final_layer_zero,
        rms_affine=not cfg.compat_checkpoint_arch,
        text_qk_norm=not cfg.compat_checkpoint_arch,
        double_qk_norm=not cfg.compat_checkpoint_arch,
        rope_style="compat" if cfg.compat_checkpoint_arch else "jax",
        attention_impl=cfg.attention_impl,
    )


def build_text_encoder(cfg: TrainConfig, device: torch.device):
    token = read_token(cfg.hf_token_file)
    enc = T5EncoderModel.from_pretrained(cfg.t5_name, token=token)
    enc.eval().to(device)
    if cfg.freeze_t5:
        for p in enc.parameters():
            p.requires_grad_(False)
    return enc


def prepare_images(pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = pixel_values.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.to(dtype=torch.float32).mul_(1.0 / 127.5).add_(-1.0)
    return images


@torch.no_grad()
def encode_text(text_encoder, input_ids, attention_mask, dtype: torch.dtype) -> torch.Tensor:
    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state.to(dtype=dtype)


def maybe_wandb(cfg: TrainConfig):
    if not is_main() or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        if "WANDB_API_KEY" not in os.environ:
            key_path = WANDB_API_KEY_FILE
            if Path(key_path).exists():
                os.environ["WANDB_API_KEY"] = Path(key_path).read_text().strip()
        import wandb

        return wandb.init(project=cfg.project, name=cfg.run_name, config=cfg.to_dict())
    except Exception as e:
        print(f"[wandb] disabled: {e}", flush=True)
        return None


def save_checkpoint(cfg, model, ema_model, opt, step):
    if not is_main():
        return
    src = model.module if hasattr(model, "module") else model
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{step:06d}.pt"
    atomic_torch_save(
        {
            "step": step,
            "model": src.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg.to_dict(),
        },
        path,
    )
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    for old in ckpts[:-cfg.save_keep]:
        old.unlink(missing_ok=True)


def latest_checkpoint(cfg: TrainConfig) -> Path | None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return ckpts[-1] if ckpts else None


def resolve_resume_checkpoint(cfg: TrainConfig) -> Path | None:
    if cfg.resume_from:
        if cfg.resume_from == "latest":
            return latest_checkpoint(cfg)
        return Path(cfg.resume_from)
    if cfg.auto_resume:
        return latest_checkpoint(cfg)
    return None


def load_checkpoint_if_available(cfg, model, ema_model, device) -> tuple[int, dict | None]:
    path = resolve_resume_checkpoint(cfg)
    if path is None:
        return 0, None
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    ema_model.load_state_dict(ckpt["ema"], strict=True)
    step = int(ckpt["step"])
    if is_main():
        print(f"[resume] loaded {path} at step={step}", flush=True)
    return step, ckpt.get("optimizer")


@torch.no_grad()
def run_visualize(cfg, model, text_encoder, batch, device, dtype, step, name="sample"):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    mask = batch["attention_mask"].to(device, non_blocking=True)
    text = encode_text(text_encoder, input_ids, mask, dtype)
    src = model.module if hasattr(model, "module") else model
    samples = euler_sample(src, text[:4], mask[:4], cfg.image_size, steps=min(cfg.n_T, 8), cfg_scale=cfg.cfg_scale)
    out = Path(cfg.output_dir) / "samples" / f"{name}_{step:06d}.png"
    if is_main():
        save_grid(samples, str(out), nrow=2)
    return samples, out


@torch.no_grad()
def run_visualize_captions(cfg, model, text_encoder, captions, device, dtype, step, name="sample"):
    from transformers import AutoTokenizer

    token = read_token(cfg.hf_token_file)
    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name, token=token, model_max_length=cfg.prompt_length)
    tok = tokenizer(
        captions[:4],
        max_length=cfg.prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device)
    mask = tok["attention_mask"].to(device)
    text = encode_text(text_encoder, input_ids, mask, dtype)
    src = model.module if hasattr(model, "module") else model
    samples = euler_sample(src, text, mask, cfg.image_size, steps=min(cfg.n_T, 8), cfg_scale=cfg.cfg_scale)
    out = Path(cfg.output_dir) / "samples" / f"{name}_{step:06d}.png"
    if is_main():
        save_grid(samples, str(out), nrow=2)
    return samples, out


@torch.no_grad()
def run_fid_eval(cfg, model, text_encoder, device, dtype, step, name="fid"):
    from transformers import AutoTokenizer

    token = read_token(cfg.hf_token_file)
    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name, token=token, model_max_length=cfg.prompt_length)
    captions = load_captions(cfg.mscoco_caption_file, limit=cfg.fid_num_samples)
    local_captions = captions[rank() :: world_size()]
    src = model.module if hasattr(model, "module") else model
    feat_model = InceptionFeatures().to(device)
    all_features = []
    first_images = None
    for start in range(0, len(local_captions), cfg.fid_batch_size):
        cap_batch = local_captions[start : start + cfg.fid_batch_size]
        tok = tokenizer(
            cap_batch,
            max_length=cfg.prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        text = encode_text(text_encoder, input_ids, mask, dtype)
        images = euler_sample(src, text, mask, cfg.image_size, steps=min(cfg.n_T, 4 if cfg.fid_num_samples <= 16 else cfg.n_T), cfg_scale=cfg.cfg_scale)
        if first_images is None and is_main():
            first_images = images[:4].detach().cpu()
        all_features.append(extract_features(feat_model, images, device))
        if is_main() and (start // cfg.fid_batch_size + 1) % 10 == 0:
            generated = min(start + cfg.fid_batch_size, len(local_captions))
            print(f"[fid] name={name} step={step} rank0_generated={generated}/{len(local_captions)} total={len(captions)}", flush=True)
    if all_features:
        local_features = np.concatenate(all_features, axis=0)
    else:
        local_features = np.empty((0, 2048), dtype=np.float32)
    if dist.is_initialized():
        gathered = [None for _ in range(world_size())] if is_main() else None
        dist.gather_object(local_features, gathered, dst=0)
        if not is_main():
            return None, None
        features = np.concatenate(gathered, axis=0)
    else:
        features = local_features
    out = None
    if first_images is not None:
        out = Path(cfg.output_dir) / "samples" / f"{name}_{step:06d}.png"
        save_grid(first_images, str(out), nrow=2)
    fid = fid_from_features(features, cfg.mscoco_stats_file)
    print(f"[fid] name={name} step={step} samples={features.shape[0]} fid={fid:.4f}", flush=True)
    return fid, out


def wandb_log_image(wb, key: str, path: Path | None, step: int):
    if wb is None or path is None or not is_main() or not path.exists():
        return
    try:
        import wandb

        wb.log({key: wandb.Image(str(path))}, step=step)
    except Exception as e:
        print(f"[wandb] image log failed for {path}: {e}", flush=True)


def train(cfg: TrainConfig, eval_only: bool = False, eval_step: int | None = None, eval_target: str = "both"):
    cfg.resolve()
    if not eval_only:
        check_training_data(cfg)
    device, local_rank = init_dist()
    seed_all(cfg.seed)
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    text_encoder = build_text_encoder(cfg, device)
    model = build_model(cfg).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    start_step, opt_state = load_checkpoint_if_available(cfg, model, ema_model, device)
    if cfg.compile_model:
        model = torch.compile(model)
    if world_size() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        betas=(0.9, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    wb = maybe_wandb(cfg)

    if eval_only:
        step = eval_step if eval_step is not None else start_step
        ema_tag = f"ema_{cfg.ema_decay:g}"
        captions = load_captions(cfg.mscoco_caption_file, limit=max(4, cfg.fid_num_samples))
        raw_sample_path = None
        ema_sample_path = None
        fid, raw_fid_path = (None, None)
        fid_ema, ema_fid_path = (None, None)
        if eval_target in ("both", "model"):
            _, raw_sample_path = run_visualize_captions(cfg, model, text_encoder, captions, device, dtype, step, name="sample")
            fid, raw_fid_path = run_fid_eval(cfg, model, text_encoder, device, dtype, step, name="fid")
        if eval_target in ("both", "ema"):
            _, ema_sample_path = run_visualize_captions(cfg, ema_model, text_encoder, captions, device, dtype, step, name="sample_ema")
            fid_ema, ema_fid_path = run_fid_eval(cfg, ema_model, text_encoder, device, dtype, step, name="fid_ema")
        if wb and fid is not None:
            wb.log({"FID/mscoco/online/cfg2.0": fid}, step=step)
        if wb and fid_ema is not None:
            wb.log({f"FID/mscoco/{ema_tag}/cfg2.0": fid_ema}, step=step)
        wandb_log_image(wb, "sample", raw_sample_path, step)
        wandb_log_image(wb, f"sample_{ema_tag}", ema_sample_path, step)
        wandb_log_image(wb, "fid_samples", raw_fid_path, step)
        wandb_log_image(wb, f"fid_samples_{ema_tag}", ema_fid_path, step)
        if wb:
            wb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    loader = make_loader(cfg)
    ema_tag = f"ema_{cfg.ema_decay:g}"

    start = time.time()
    last_log_time = start
    last_log_step = start_step
    accum_loss = 0.0
    accum_loss_tensor = torch.zeros((), device=device)
    accum_loss_count = 0
    accum_timing = {k: 0.0 for k in ("data_s", "h2d_s", "t5_s", "train_s", "opt_s")}
    opt.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    for step in range(start_step, cfg.num_steps):
        step_t_data = 0.0
        step_t_h2d = 0.0
        step_t_t5 = 0.0
        step_t_train = 0.0
        for accum in range(cfg.grad_accum_steps):
            t0 = time.time()
            batch = next(data_iter)
            t1 = time.time()
            images = prepare_images(batch["pixel_values"], device)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            t2 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                text = encode_text(text_encoder, input_ids, mask, dtype)
            t3 = time.time()
            sync_context = (
                model.no_sync()
                if hasattr(model, "no_sync") and accum < cfg.grad_accum_steps - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.autocast("cuda", dtype=dtype):
                    loss, metrics, _ = training_loss(
                        model,
                        images,
                        text,
                        mask,
                        label_drop_rate=cfg.label_drop_rate,
                        t_lognorm_mu=cfg.t_lognorm_mu,
                        t_lognorm_sigma=cfg.t_lognorm_sigma,
                        noise_scale=cfg.noise_scale,
                    )
                    loss = loss / cfg.grad_accum_steps
                loss.backward()
            t4 = time.time()
            step_t_data += t1 - t0
            step_t_h2d += t2 - t1
            step_t_t5 += t3 - t2
            step_t_train += t4 - t3
            if is_main() and cfg.defer_loss_sync:
                accum_loss_tensor = accum_loss_tensor + metrics["loss"].detach()
                accum_loss_count += 1
            elif is_main():
                accum_loss += float(metrics["loss"].detach().cpu())
                accum_loss_count += 1

        opt_t0 = time.time()
        lr = jax_warmup_constant_lr(step, cfg.num_steps, cfg.warmup_steps, cfg.learning_rate)
        for group in opt.param_groups:
            group["lr"] = lr
        parameters = [p for p in model.parameters() if p.grad is not None]
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
        else:
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2) for p in parameters]),
                2,
            ) if parameters else torch.zeros((), device=device)
        opt.step()
        opt.zero_grad(set_to_none=True)
        update_ema(ema_model, model, cfg.ema_decay)
        torch.cuda.synchronize(device)
        opt_t1 = time.time()
        accum_timing["data_s"] += step_t_data
        accum_timing["h2d_s"] += step_t_h2d
        accum_timing["t5_s"] += step_t_t5
        accum_timing["train_s"] += step_t_train
        accum_timing["opt_s"] += opt_t1 - opt_t0

        if is_main() and (step + 1) % cfg.log_every == 0:
            now = time.time()
            elapsed = now - start
            steps_this_run = step + 1 - start_step
            interval_steps = step + 1 - last_log_step
            interval_elapsed = now - last_log_time
            sps = steps_this_run / max(elapsed, 1e-6)
            log = {
                "loss": (float(accum_loss_tensor.detach().cpu()) if cfg.defer_loss_sync else accum_loss)
                / max(accum_loss_count, 1),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "steps_per_second": sps,
                "interval_steps_per_second": interval_steps / max(interval_elapsed, 1e-6),
                "elapsed_s": elapsed,
                "step": step + 1,
            }
            for key, value in accum_timing.items():
                log[key] = value / max(interval_steps, 1)
            print(json.dumps(log), flush=True)
            if wb:
                wb.log(log, step=step + 1)
            accum_loss = 0.0
            accum_loss_tensor = torch.zeros((), device=device)
            accum_loss_count = 0
            accum_timing = {k: 0.0 for k in accum_timing}
            last_log_time = now
            last_log_step = step + 1

        if (step + 1) % cfg.ckpt_every == 0:
            save_checkpoint(cfg, model, ema_model, opt, step + 1)
        if (step + 1) % cfg.sample_every == 0:
            _, raw_sample_path = run_visualize(cfg, model, text_encoder, batch, device, dtype, step + 1, name="sample")
            _, ema_sample_path = run_visualize(cfg, ema_model, text_encoder, batch, device, dtype, step + 1, name="sample_ema")
            wandb_log_image(wb, "sample", raw_sample_path, step + 1)
            wandb_log_image(wb, f"sample_{ema_tag}", ema_sample_path, step + 1)
        if (step + 1) % cfg.fid_every == 0:
            fid, raw_fid_path = run_fid_eval(cfg, model, text_encoder, device, dtype, step + 1, name="fid")
            fid_ema, ema_fid_path = run_fid_eval(cfg, ema_model, text_encoder, device, dtype, step + 1, name="fid_ema")
            if wb and fid is not None:
                wb.log({"FID/mscoco/online/cfg2.0": fid}, step=step + 1)
            if wb and fid_ema is not None:
                wb.log({f"FID/mscoco/{ema_tag}/cfg2.0": fid_ema}, step=step + 1)
            wandb_log_image(wb, "fid_samples", raw_fid_path, step + 1)
            wandb_log_image(wb, f"fid_samples_{ema_tag}", ema_fid_path, step + 1)
    if wb:
        wb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--sync-loss-every-step", action="store_true")
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--ckpt-every", type=int)
    parser.add_argument("--fid-every", type=int)
    parser.add_argument("--fid-num-samples", type=int)
    parser.add_argument("--fid-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dataloader-prefetch", type=int)
    parser.add_argument("--dataloader-multiprocessing-context", choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--shuffle-buffer", type=int)
    parser.add_argument(
        "--dataset-backend",
        choices=[
            "local_folder",
            "finetune_wds",
        ],
    )
    parser.add_argument("--local-dataset-dir")
    parser.add_argument("--finetune-dataset-dir")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-step", type=int)
    parser.add_argument("--eval-target", choices=["both", "model", "ema"], default="both")
    parser.add_argument("--compat-checkpoint-arch", action="store_true")
    parser.add_argument("--attention-impl", choices=["einsum", "sdpa"])
    args = parser.parse_args()
    cfg = TrainConfig()
    if args.config_file:
        cfg.update_from_yaml(args.config_file)
    cfg.resolve()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.run_name:
        cfg.run_name = args.run_name
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.micro_batch_size is not None:
        cfg.micro_batch_size = args.micro_batch_size
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.no_auto_resume:
        cfg.auto_resume = False
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.sync_loss_every_step:
        cfg.defer_loss_sync = False
    if args.sample_every is not None:
        cfg.sample_every = args.sample_every
    if args.ckpt_every is not None:
        cfg.ckpt_every = args.ckpt_every
    if args.fid_every is not None:
        cfg.fid_every = args.fid_every
    if args.fid_num_samples is not None:
        cfg.fid_num_samples = args.fid_num_samples
    if args.fid_batch_size is not None:
        cfg.fid_batch_size = args.fid_batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.dataloader_prefetch is not None:
        cfg.dataloader_prefetch = args.dataloader_prefetch
    if args.dataloader_multiprocessing_context is not None:
        cfg.dataloader_multiprocessing_context = args.dataloader_multiprocessing_context
    if args.shuffle_buffer is not None:
        cfg.shuffle_buffer = args.shuffle_buffer
    if args.dataset_backend is not None:
        cfg.dataset_backend = args.dataset_backend
    if args.local_dataset_dir is not None:
        cfg.local_dataset_dir = args.local_dataset_dir
    if args.finetune_dataset_dir is not None:
        cfg.finetune_dataset_dir = args.finetune_dataset_dir
    if args.compat_checkpoint_arch:
        cfg.compat_checkpoint_arch = True
    if args.attention_impl is not None:
        cfg.attention_impl = args.attention_impl
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    cfg.resolve()
    train(cfg, eval_only=args.eval_only, eval_step=args.eval_step, eval_target=args.eval_target)


if __name__ == "__main__":
    main()
