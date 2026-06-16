from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def seed_all(seed: int) -> None:
    seed = seed + rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_token(path: str) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text().strip()


def amp_dtype(name: str) -> torch.dtype:
    if name.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name.lower() in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def jax_warmup_constant_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    del total_steps
    if step < warmup_steps:
        start_lr = 1e-6
        alpha = float(step + 1) / max(warmup_steps, 1)
        return start_lr + alpha * (base_lr - start_lr)
    return base_lr


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    src = model.module if hasattr(model, "module") else model
    for ema_p, p in zip(ema_model.parameters(), src.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), src.buffers()):
        ema_b.copy_(b)


def atomic_torch_save(obj: object, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as zf:
        if not any(name.endswith("/.data/serialization_id") for name in zf.namelist()):
            raise RuntimeError(f"checkpoint archive is missing serialization metadata: {tmp}")
    tmp.replace(path)
    dir_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
