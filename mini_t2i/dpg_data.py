from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


def _category_tuple(question: dict[str, Any]) -> str:
    broad = str(question.get("category_broad") or "").strip()
    detailed = str(question.get("category_detailed") or "").strip()
    label = " - ".join(part for part in (broad, detailed) if part)
    text = str(question.get("question") or "").strip()
    return f"{label} ({text})" if label else text


def _expand_hf_rows(rows) -> list[dict[str, Any]]:
    expanded = []
    for row in rows:
        item_id = row["item_id"]
        prompt = row["prompt"]
        for question in row["questions"]:
            expanded.append(
                {
                    "item_id": item_id,
                    "text": prompt,
                    "proposition_id": int(question["proposition_id"]),
                    "dependency": str(question.get("dependency", "0")),
                    "category_broad": question.get("category_broad", ""),
                    "category_detailed": question.get("category_detailed", ""),
                    "tuple": _category_tuple(question),
                    "question_natural_language": question["question"],
                }
            )
    return expanded


def _read_local_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    if suffix == ".parquet":
        data = pd.read_parquet(path).to_dict("records")
        if data and "questions" in data[0]:
            return _expand_hf_rows(data)
        return data
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        if data and "questions" in data[0]:
            return _expand_hf_rows(data)
        return data
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("rows", []))
        if data and "questions" in data[0]:
            return _expand_hf_rows(data)
        return data
    raise ValueError(f"unsupported DPG-Bench data file format: {path}")


def load_dpg_rows(source: str | os.PathLike[str], split: str = "test") -> list[dict[str, Any]]:
    source_str = str(source)
    path = Path(source_str)
    if path.exists():
        return _read_local_rows(path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"DPG-Bench data source {source_str!r} is not a local file. "
            "Install `datasets` to load it from Hugging Face, or pass a local CSV/parquet/JSONL file."
        ) from exc

    dataset = load_dataset(source_str, split=split)
    return _expand_hf_rows(dataset)


def load_dpg_prompts(source: str | os.PathLike[str], split: str = "test") -> list[tuple[str, str]]:
    prompts: OrderedDict[str, str] = OrderedDict()
    for row in load_dpg_rows(source, split=split):
        prompts.setdefault(row["item_id"], row["text"])
    return list(prompts.items())
