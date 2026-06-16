from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def save_grid(images: torch.Tensor, path: str, nrow: int = 4) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images = (images.detach().float().cpu().clamp(-1, 1) + 1) / 2
    save_image(make_grid(images, nrow=nrow), path)

