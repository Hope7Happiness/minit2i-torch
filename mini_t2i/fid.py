from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg


class InceptionFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from pytorch_fid.inception import InceptionV3

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.net = InceptionV3([block_idx], normalize_input=True, resize_input=True).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1, 1) + 1) / 2
        out = self.net(x)[0]
        return out.squeeze(-1).squeeze(-1)


@torch.no_grad()
def extract_features(model: InceptionFeatures, images: torch.Tensor, device: torch.device) -> np.ndarray:
    feats = []
    for start in range(0, images.shape[0], 16):
        feats.append(model(images[start : start + 16].to(device)).float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def load_captions(path: str, limit: int | None = None) -> list[str]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        vals = list(data.values())
    else:
        vals = data
    caps = []
    for x in vals:
        if isinstance(x, str):
            caps.append(x)
        elif isinstance(x, dict):
            caps.append(x.get("caption") or x.get("text") or x.get("prompt") or "")
        else:
            caps.append(str(x))
        if limit is not None and len(caps) >= limit:
            break
    return caps


def compute_fid_from_images(images: torch.Tensor, stats_file: str, device: torch.device) -> float:
    feat_model = InceptionFeatures().to(device)
    feats = extract_features(feat_model, images, device)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    stats = np.load(stats_file)
    return frechet_distance(mu, sigma, stats["mu"], stats["sigma"])


def fid_from_features(features: np.ndarray, stats_file: str) -> float:
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    stats = np.load(stats_file)
    return frechet_distance(mu, sigma, stats["mu"], stats["sigma"])
