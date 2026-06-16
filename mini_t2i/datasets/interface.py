from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import IterableDataset

from ..config import TrainConfig
from ..utils import rank, world_size
from .finetune import FinetuneMixedStream


class DatasetProvider(Protocol):
    def build(self) -> IterableDataset:
        ...


class LocalTensorChunkStream(IterableDataset):
    """Local-only path for preprocessed tensor chunks.

    This intentionally assumes all dataset assets already exist under a local
    directory. It performs no downloads and no image decoding. The expected file
    format matches the local tensor chunks used by the pretraining pipeline:
    `chunk_*.pt` files containing `pixel_values`, `input_ids`,
    `attention_mask`, and optional `caption`.
    """

    def __init__(self, cfg: TrainConfig):
        if not cfg.local_dataset_dir:
            raise RuntimeError("local_dataset_dir is required for local_folder backend")
        self.cfg = cfg
        self.root = Path(cfg.local_dataset_dir)
        self.files = sorted(self.root.glob("chunk_*.pt"))
        if not self.files:
            raise RuntimeError(f"no local tensor chunks found in {self.root}")

    def _files_for_worker(self) -> list[Path]:
        worker = torch.utils.data.get_worker_info()
        shard_index = rank()
        num_shards = world_size()
        if worker is not None:
            shard_index = rank() * worker.num_workers + worker.id
            num_shards = world_size() * worker.num_workers
        return self.files[shard_index::num_shards]

    def __iter__(self):
        files = self._files_for_worker()
        if not files:
            raise RuntimeError(
                f"local dataset has {len(self.files)} chunks, fewer than rank/worker shards "
                f"({world_size()} x workers)"
            )
        rng = random.Random(self.cfg.seed + rank())
        while True:
            order = list(files)
            rng.shuffle(order)
            for path in order:
                chunk = torch.load(path, map_location="cpu", weights_only=False)
                n = int(chunk["pixel_values"].shape[0])
                indices = list(range(n))
                rng.shuffle(indices)
                captions = chunk.get("caption") or [""] * n
                for idx in indices:
                    yield {
                        "pixel_values": chunk["pixel_values"][idx],
                        "input_ids": chunk["input_ids"][idx].long(),
                        "attention_mask": chunk["attention_mask"][idx].long(),
                        "caption": captions[idx],
                    }


class LocalFolderDatasetProvider:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def build(self) -> IterableDataset:
        return LocalTensorChunkStream(self.cfg)


class FinetuneWDSDatasetProvider:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def build(self) -> IterableDataset:
        return FinetuneMixedStream(self.cfg)


def provider_for_config(cfg: TrainConfig) -> DatasetProvider:
    if cfg.dataset_backend == "local_folder":
        return LocalFolderDatasetProvider(cfg)
    if cfg.dataset_backend == "finetune_wds":
        return FinetuneWDSDatasetProvider(cfg)
    raise ValueError(
        f"unknown dataset_backend={cfg.dataset_backend!r}; supported backends are "
        "'local_folder' and 'finetune_wds'"
    )
