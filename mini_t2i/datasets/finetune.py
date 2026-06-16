from __future__ import annotations

import io
import json
import random
import shlex
from pathlib import Path
from typing import Iterator

import torch
import webdataset as wds
from huggingface_hub import hf_hub_download, hf_hub_url
from PIL import Image, ImageFile
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from torchvision import transforms
from transformers import AutoTokenizer

from ..config import TrainConfig
from ..utils import rank, read_token, world_size

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_hf_dataset(*args, **kwargs):
    # When training is launched as `python mini_t2i/train.py`, sys.path[0] is
    # `mini_t2i/`, so a top-level `import datasets` resolves to our local
    # package. Import HF datasets lazily only for the online fallback.
    import importlib
    import sys

    package_dir = str(Path(__file__).resolve().parents[1])
    removed = []
    while package_dir in sys.path:
        sys.path.remove(package_dir)
        removed.append(package_dir)
    shadow = sys.modules.get("datasets")
    if shadow is not None and str(getattr(shadow, "__file__", "")).startswith(package_dir):
        del sys.modules["datasets"]
    try:
        return importlib.import_module("datasets").load_dataset(*args, **kwargs)
    finally:
        sys.path[:0] = removed


def _check_pillow_runtime() -> None:
    version = tuple(int(part) for part in Image.__version__.split(".")[:2])
    if version < (12, 2):
        raise RuntimeError(
            f"fine-tuning image decoding requires Pillow>=12.2.0; found Pillow {Image.__version__}"
        )


def _curl_pipe_url(url: str) -> str:
    return (
        "pipe:curl -L -s -f --connect-timeout 5 --max-time 120 "
        f"--retry 2 --retry-delay 1 {shlex.quote(url)}"
    )


def _deepfusion_jpeg_roundtrip(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    converted = Image.open(buf).convert("RGB")
    converted.load()
    return converted


BLIP3_FT60K_TARS = [
    "dalle3.tar",
    "geneval_train.tar",
    "human_gestures.tar",
    "journeyDB.tar",
    "mscoco_human.tar",
    "object_1.tar",
    "object_2.tar",
    "occupation_1.tar",
    "occupation_2.tar",
    "text_1.tar",
    "text_2.tar",
]


def source_files_for_root(root: Path, source: str) -> list[Path]:
    source_dir = root / source
    base = source_dir if source_dir.exists() else root
    if source == "blip3_ft60k":
        return [base / name for name in BLIP3_FT60K_TARS if (base / name).exists()]
    if source == "dalle3":
        return sorted(base.glob("shard-*.tar"))
    if source == "sharegpt4o":
        files = sorted(base.glob("shard-*.tar"))
        return files or sorted(base.glob("text_to_image_part_*.tar"))
    raise ValueError(f"unknown finetune source: {source}")


def _identity_split(src):
    yield from src


def _urls_for_local_worker(urls: list[str]) -> list[str]:
    if not urls:
        return urls
    worker = torch.utils.data.get_worker_info()
    worker_id = worker.id if worker is not None else 0
    num_workers = worker.num_workers if worker is not None else 1
    shard_index = rank() * num_workers + worker_id
    num_shards = world_size() * num_workers
    selected = urls[shard_index::num_shards]
    return selected or [urls[shard_index % len(urls)]]


def _worker_rng(cfg: TrainConfig, seed_offset: int = 0) -> random.Random:
    worker = torch.utils.data.get_worker_info()
    worker_id = worker.id if worker is not None else 0
    num_workers = worker.num_workers if worker is not None else 1
    return random.Random(cfg.seed + seed_offset + rank() * max(1, num_workers) + worker_id)


def _choose_weighted(rng: random.Random, weights: list[float]) -> int:
    total = float(sum(weights))
    if total <= 0:
        raise RuntimeError("finetune mix weights must sum to a positive value")
    pick = rng.random() * total
    accum = 0.0
    for idx, weight in enumerate(weights):
        accum += float(weight)
        if pick < accum:
            return idx
    return len(weights) - 1


def _yield_weighted_forever(make_source, count: int, weights: list[float], cfg: TrainConfig):
    rng = _worker_rng(cfg, seed_offset=1009)
    sources = [iter(make_source(idx)) for idx in range(count)]
    while True:
        idx = _choose_weighted(rng, weights)
        for _ in range(count):
            try:
                yield next(sources[idx])
                break
            except StopIteration:
                sources[idx] = iter(make_source(idx))
                try:
                    yield next(sources[idx])
                    break
                except StopIteration:
                    idx = (idx + 1) % count
        else:
            raise RuntimeError("all finetune data sources are empty")


class FinetuneWDSStream(IterableDataset):
    """Local WebDataset reader for the JAX fine-tuning data mixture.

    The upstream ablation config mixes three dataset groups:
    BLIP3o-60k, DALL-E3, and ShareGPT4o. This class keeps that grouping intact
    so the configured mix weights match JAX semantics.
    """

    def __init__(self, cfg: TrainConfig):
        _check_pillow_runtime()
        if not cfg.finetune_dataset_dir:
            raise RuntimeError("finetune_dataset_dir is required for finetune_wds backend")
        self.cfg = cfg
        self.root = Path(cfg.finetune_dataset_dir)
        self.token = read_token(cfg.hf_token_file)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.t5_name,
            token=self.token,
            model_max_length=cfg.prompt_length,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(cfg.image_size),
                transforms.PILToTensor(),
            ]
        )
        self.groups = self._resolve_groups()
        if len(self.cfg.finetune_mix_weights) != len(self.groups):
            raise RuntimeError(
                "finetune_mix_weights length must match resolved finetune source groups: "
                f"{len(self.cfg.finetune_mix_weights)} weights for {len(self.groups)} groups"
            )

    def _resolve_groups(self) -> list[list[str]]:
        groups = []
        missing = []
        for source in self.cfg.finetune_sources:
            files = self._source_files(source)
            if files:
                groups.append([str(path) for path in files])
            else:
                missing.append(source)
        if missing:
            raise RuntimeError(
                f"missing finetune source(s) under {self.root}: {', '.join(missing)}. "
                "Expected local WebDataset tar files; this backend does not download gs:// data."
            )
        return groups

    def _source_files(self, source: str) -> list[Path]:
        source_dir = self.root / source
        base = source_dir if source_dir.exists() else self.root
        if source == "blip3_ft60k":
            return [base / name for name in BLIP3_FT60K_TARS if (base / name).exists()]
        if source == "dalle3":
            return sorted(base.glob("shard-*.tar"))
        if source == "sharegpt4o":
            files = sorted(base.glob("shard-*.tar"))
            return files or sorted(base.glob("text_to_image_part_*.tar"))
        raise ValueError(f"unknown finetune source: {source}")

    def _make_dataset(self, urls: list[str], seed_offset: int, source: str | None = None):
        local_urls = _urls_for_local_worker(urls)
        # ResampledShards can block indefinitely on local HF-cache symlinks in
        # our WebDataset version. Local staged tar files are already shuffled at
        # the sample/shard level, so repeat the plain WebDataset stream instead.
        return (
            wds.WebDataset(
                local_urls,
                handler=wds.warn_and_continue,
                empty_check=False,
                shardshuffle=max(1, len(local_urls)),
                nodesplitter=_identity_split,
                workersplitter=_identity_split,
                repeat=True,
            )
            .shuffle(self.cfg.shuffle_buffer, rng=random.Random(self.cfg.seed + seed_offset))
            .decode("pil", handler=wds.warn_and_continue)
            .map(lambda sample: self._process_sample(sample, source=source), handler=wds.warn_and_continue)
        )

    def _process_sample(self, sample: dict, source: str | None = None) -> dict[str, torch.Tensor | str]:
        image = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
        if image is None:
            raise ValueError(f"missing image for finetune WDS sample {sample.get('__key__')}")
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected decoded PIL image, got {type(image)!r}")
        caption = sample.get("txt", "")
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8", errors="replace")
        caption = str(caption)
        tok = self.tokenizer(
            caption,
            max_length=self.cfg.prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": self.transform(image.convert("RGB")),
            "input_ids": tok["input_ids"][0].long(),
            "attention_mask": tok["attention_mask"][0].long(),
            "caption": caption,
            "source": source or "",
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        yield from _yield_weighted_forever(
            lambda idx: self._make_dataset(self.groups[idx], idx, source=self.cfg.finetune_sources[idx]),
            len(self.groups),
            self.cfg.finetune_mix_weights,
            self.cfg,
        )


class FinetuneMixedStream(FinetuneWDSStream):
    """Fine-tuning mixture with local public-data fallbacks.

    JAX mixes BLIP3o-60k WebDataset shards, DALL-E3 rows, and ShareGPT4o
    text-to-image tar parts. Our staged reproduction data keeps DALL-E3 as HF
    parquet files and ShareGPT4o captions in the dataset JSON, so the release
    backend accepts both pure local WebDataset shards and this staged format.
    """

    def __init__(self, cfg: TrainConfig):
        _check_pillow_runtime()
        self.cfg = cfg
        self.root = Path(cfg.finetune_dataset_dir) if cfg.finetune_dataset_dir else None
        self.token = read_token(cfg.hf_token_file)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.t5_name,
            token=self.token,
            model_max_length=cfg.prompt_length,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(cfg.image_size),
                transforms.PILToTensor(),
            ]
        )
        self.group_specs = [self._source_spec(source) for source in cfg.finetune_sources]
        if len(self.cfg.finetune_mix_weights) != len(self.group_specs):
            raise RuntimeError(
                "finetune_mix_weights length must match finetune_sources: "
                f"{len(self.cfg.finetune_mix_weights)} weights for {len(self.group_specs)} sources"
            )

    def _source_spec(self, source: str):
        files = self._source_files(source) if self.root is not None else []
        if source == "blip3_ft60k":
            if files:
                return ("local_wds", source, [str(path) for path in files])
            urls = [
                _curl_pipe_url(
                    hf_hub_url(
                        repo_id=self.cfg.finetune_blip3o_repo,
                        filename=filename,
                        repo_type="dataset",
                    )
                )
                for filename in BLIP3_FT60K_TARS
            ]
            return ("local_wds", source, urls)
        if source == "dalle3":
            if files:
                return ("local_wds", source, [str(path) for path in files])
            parquet_files = self._dalle3_parquet_files()
            if parquet_files:
                return ("hf_dalle3_local", source, [str(path) for path in parquet_files])
            return ("hf_dalle3", source, self.cfg.finetune_dalle3_repo)
        if source == "sharegpt4o":
            urls = [str(path) for path in files] if files else [
                _curl_pipe_url(
                    hf_hub_url(
                        repo_id=self.cfg.finetune_sharegpt4o_repo,
                        filename=f"text_to_image_part_{idx}.tar",
                        repo_type="dataset",
                    )
                )
                for idx in range(int(self.cfg.finetune_sharegpt4o_parts))
            ]
            return ("hf_sharegpt4o", source, urls)
        raise RuntimeError(
            f"missing local source {source!r}. Public HF fallback is only defined for dalle3/sharegpt4o."
        )

    def _source_files(self, source: str) -> list[Path]:
        if self.root is None:
            return []
        return super()._source_files(source)

    def _dalle3_parquet_files(self) -> list[Path]:
        if self.root is None:
            return []
        return sorted((self.root / "dalle3_parquet").glob("*.parquet"))

    def _make_dataset_from_spec(self, spec, seed_offset: int):
        kind, source, payload = spec
        if kind == "local_wds":
            return self._make_dataset(payload, seed_offset, source=source)
        if kind == "hf_dalle3_local":
            return HFDalle3Stream(self.cfg, self.tokenizer, self.transform, data_files=payload)
        if kind == "hf_dalle3":
            return HFDalle3Stream(self.cfg, self.tokenizer, self.transform)
        if kind == "hf_sharegpt4o":
            return HFShareGPT4oTextToImageStream(self.cfg, self.tokenizer, self.transform, payload)
        raise ValueError(f"unknown finetune dataset spec kind={kind!r} source={source!r}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        yield from _yield_weighted_forever(
            lambda idx: self._make_dataset_from_spec(self.group_specs[idx], idx),
            len(self.group_specs),
            self.cfg.finetune_mix_weights,
            self.cfg,
        )


class HFDalle3Stream(IterableDataset):
    def __init__(self, cfg: TrainConfig, tokenizer: AutoTokenizer, transform, data_files: list[str] | None = None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.transform = transform
        self.data_files = data_files

    def __iter__(self):
        while True:
            if self.data_files:
                yield from self._iter_local_parquet()
                continue
            ds = _load_hf_dataset(
                self.cfg.finetune_dalle3_repo,
                split="train",
                streaming=bool(self.cfg.finetune_hf_streaming),
            )
            yield from _iter_hf_rows(ds, self.cfg, self._process_row)

    def _iter_local_parquet(self):
        files = _urls_for_local_worker([str(path) for path in self.data_files or []])
        if not files:
            raise RuntimeError("DALL-E3 local parquet reader got no files")
        rng = _worker_rng(self.cfg, seed_offset=2027)
        while True:
            rng.shuffle(files)
            for filename in files:
                parquet = pq.ParquetFile(filename)
                row_groups = list(range(parquet.num_row_groups))
                rng.shuffle(row_groups)
                for row_group in row_groups:
                    columns = ["caption", "synthetic_caption", "image"]
                    table = parquet.read_row_group(row_group, columns=columns)
                    rows = table.to_pylist()
                    rng.shuffle(rows)
                    for row in rows:
                        try:
                            yield self._process_row(row)
                        except Exception:
                            continue

    def _process_row(self, row: dict) -> dict[str, torch.Tensor | str]:
        image = row["image"]
        if isinstance(image, dict):
            image_bytes = image.get("bytes")
            if image_bytes is None:
                raise ValueError("DALL-E3 row image dict has no bytes field")
            image = Image.open(io.BytesIO(image_bytes))
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected PIL image from DALL-E3 HF dataset, got {type(image)!r}")
        caption = str(row.get("caption") or row.get("synthetic_caption") or "").strip() + "\n"
        item = _tokenize_image_caption(self.cfg, self.tokenizer, self.transform, image, caption)
        item["source"] = "dalle3"
        return item


class HFShareGPT4oTextToImageStream(IterableDataset):
    def __init__(self, cfg: TrainConfig, tokenizer: AutoTokenizer, transform, urls: list[str]):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.transform = transform
        self.urls = urls
        self.caption_by_key = self._load_captions()

    def _load_captions(self) -> dict[str, str]:
        local_path = self.root_json_path()
        if local_path is not None:
            path = local_path
        else:
            path = Path(
                hf_hub_download(
                    repo_id=self.cfg.finetune_sharegpt4o_repo,
                    filename="text_to_image.json",
                    repo_type="dataset",
                )
            )
        rows = json.load(open(path, "r", encoding="utf-8"))
        filter_tokenizer = AutoTokenizer.from_pretrained(self.cfg.finetune_sharegpt4o_filter_tokenizer)
        captions = {}
        for row in rows:
            image_path = str(row.get("output_image") or "")
            key = str(Path(image_path).with_suffix(""))
            caption = str(row.get("input_prompt") or "")
            tokens = filter_tokenizer(caption, truncation=False, return_tensors="pt", verbose=False)
            if int(tokens.input_ids.shape[1]) <= int(self.cfg.finetune_sharegpt4o_max_tokens):
                captions[key] = caption + "\n"
        return captions

    def root_json_path(self) -> Path | None:
        root = Path(self.cfg.finetune_dataset_dir) if self.cfg.finetune_dataset_dir else None
        if root is None:
            return None
        path = root / "sharegpt4o" / "text_to_image.json"
        return path if path.exists() else None

    def __iter__(self):
        local_urls = _urls_for_local_worker(self.urls)
        ds = (
            wds.WebDataset(
                local_urls,
                handler=wds.warn_and_continue,
                empty_check=False,
                shardshuffle=max(1, len(local_urls)),
                nodesplitter=_identity_split,
                workersplitter=_identity_split,
                repeat=True,
            )
            .shuffle(self.cfg.shuffle_buffer, rng=random.Random(self.cfg.seed + 17))
            .select(self._has_caption)
            .decode("pil", handler=wds.warn_and_continue)
            .map(self._process_sample, handler=wds.warn_and_continue)
        )
        yield from ds

    def _has_caption(self, sample: dict) -> bool:
        key = str(sample.get("__key__", ""))
        return key in self.caption_by_key

    def _process_sample(self, sample: dict) -> dict[str, torch.Tensor | str]:
        key = str(sample.get("__key__", ""))
        image = sample.get("png") or sample.get("jpg") or sample.get("jpeg")
        if image is None:
            raise ValueError(f"missing image for ShareGPT4o sample {key}")
        caption = self.caption_by_key.get(key, "")
        if not caption:
            raise ValueError(f"missing caption for ShareGPT4o sample {key}")
        try:
            image = _deepfusion_jpeg_roundtrip(image)
            item = _tokenize_image_caption(self.cfg, self.tokenizer, self.transform, image, caption)
        except Exception as exc:
            raise ValueError(f"failed to process ShareGPT4o sample {key}: {exc}") from exc
        item["source"] = "sharegpt4o"
        return item


def _iter_hf_rows(ds, cfg: TrainConfig, process):
    worker = torch.utils.data.get_worker_info()
    shard_index = rank()
    num_shards = world_size()
    if worker is not None:
        shard_index = rank() * worker.num_workers + worker.id
        num_shards = world_size() * worker.num_workers
    rng = random.Random(cfg.seed + shard_index)
    buffer = []
    buffer_size = max(1, int(cfg.shuffle_buffer))
    for idx, row in enumerate(ds):
        if idx % num_shards != shard_index:
            continue
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        j = rng.randrange(len(buffer))
        try:
            yield process(buffer[j])
        except Exception:
            pass
        buffer[j] = row
    while buffer:
        j = rng.randrange(len(buffer))
        row = buffer.pop(j)
        try:
            yield process(row)
        except Exception:
            pass


def _tokenize_image_caption(cfg: TrainConfig, tokenizer, transform, image: Image.Image, caption: str):
    tok = tokenizer(
        caption,
        max_length=cfg.prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "pixel_values": transform(image.convert("RGB")),
        "input_ids": tok["input_ids"][0].long(),
        "attention_mask": tok["attention_mask"][0].long(),
        "caption": caption,
        "source": "",
    }
