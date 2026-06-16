from __future__ import annotations

import itertools
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .datasets.finetune import source_files_for_root
from .datasets.interface import provider_for_config


def _require_dir(path: str | Path | None, name: str) -> Path:
    if not path:
        raise RuntimeError(f"{name} is required for the selected dataset backend")
    root = Path(path)
    if not root.exists():
        raise RuntimeError(f"{name} does not exist: {root}")
    if not root.is_dir():
        raise RuntimeError(f"{name} must be a directory, got: {root}")
    return root


def _check_local_tensor_chunks(cfg: TrainConfig) -> None:
    root = _require_dir(cfg.local_dataset_dir, "local_dataset_dir")
    files = sorted(root.glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(
            f"no pretraining tensor chunks found in {root}. "
            "Expected files named chunk_*.pt; run tools/prepare_cc12m_chunks.py first."
        )

    first = files[0]
    try:
        sample = torch.load(first, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"failed to read pretraining tensor chunk {first}: {exc}") from exc

    required = {"pixel_values", "input_ids", "attention_mask"}
    missing = sorted(required - set(sample.keys()))
    if missing:
        raise RuntimeError(
            f"pretraining tensor chunk {first} is missing required key(s): {', '.join(missing)}"
        )


def _check_tar_has_sample(path: Path, source: str) -> None:
    if not tarfile.is_tarfile(path):
        raise RuntimeError(f"finetune source {source} has a non-tar shard: {path}")
    try:
        with tarfile.open(path) as tar:
            names = [member.name.lower() for _, member in zip(range(256), tar) if member.isfile()]
    except Exception as exc:
        raise RuntimeError(f"failed to inspect finetune tar shard {path}: {exc}") from exc

    if not names:
        raise RuntimeError(f"finetune tar shard is empty: {path}")
    has_image = any(name.endswith((".jpg", ".jpeg", ".png")) for name in names)
    has_text = any(name.endswith(".txt") for name in names)
    if not has_image or not has_text:
        raise RuntimeError(
            f"finetune tar shard {path} does not look like image/text WebDataset data "
            f"(image={has_image}, txt={has_text})"
        )


def _check_finetune_wds(cfg: TrainConfig) -> None:
    if not cfg.finetune_dataset_dir:
        if getattr(cfg, "finetune_hf_streaming", False):
            return
        raise RuntimeError("finetune_dataset_dir is required for finetune_wds backend")

    root_path = Path(cfg.finetune_dataset_dir)
    if not root_path.exists() and getattr(cfg, "finetune_hf_streaming", False):
        return

    root = _require_dir(cfg.finetune_dataset_dir, "finetune_dataset_dir")
    if not cfg.finetune_sources:
        raise RuntimeError("finetune_sources must contain at least one source")

    missing = []
    first_files = []
    for source in cfg.finetune_sources:
        files = source_files_for_root(root, source)
        if files:
            first_files.append((source, files[0]))
        elif source == "dalle3" and sorted((root / "dalle3_parquet").glob("*.parquet")):
            continue
        else:
            missing.append(source)
    if missing:
        raise RuntimeError(
            f"missing finetune source(s) under {root}: {', '.join(missing)}. "
            "Expected local WebDataset .tar shards; run tools/prepare_120k_mix_layout.py first."
        )

    source, first = first_files[0]
    _check_tar_has_sample(first, source)


def check_training_data(cfg: TrainConfig) -> None:
    """Fail early with clear messages when configured training data is absent."""
    if cfg.dataset_backend == "local_folder":
        _check_local_tensor_chunks(cfg)
        return
    if cfg.dataset_backend == "finetune_wds":
        _check_finetune_wds(cfg)
        return
    raise RuntimeError(
        f"unknown dataset_backend={cfg.dataset_backend!r}; supported backends are "
        "'local_folder' and 'finetune_wds'"
    )


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch], dim=0),
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "caption": [x["caption"] for x in batch],
        "source": [x.get("source", "") for x in batch],
    }


def make_loader(cfg: TrainConfig) -> DataLoader:
    dataset = provider_for_config(cfg).build()
    num_workers = cfg.num_workers
    multiprocessing_context = cfg.dataloader_multiprocessing_context if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=cfg.dataloader_prefetch if num_workers > 0 else None,
        multiprocessing_context=multiprocessing_context,
        drop_last=True,
    )


def take_batches(loader: DataLoader, n: int):
    return itertools.islice(loader, n)
