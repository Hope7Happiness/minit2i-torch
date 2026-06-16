from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# User settings: edit these paths for your machine.
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/path/to/dataset/root")
OUTPUT_ROOT = Path("/path/to/output/root")
CHECKPOINT_ROOT = Path("/path/to/checkpoint/root")
EVAL_OUTPUT_ROOT = Path("/path/to/evaluation/output/root")
ASSET_ROOT = Path("/path/to/evaluation/assets")

HF_TOKEN_FILE = Path("/path/to/secrets/hf_token")
WANDB_API_KEY_FILE = Path("/path/to/secrets/wandb_api_key")

DPG_BENCH_DATA = "Jialuo21/DPG-Bench"

PYTHON = "python"
GENEVAL_PYTHON = "python"
DPG_PYTHON = "python"


# ---------------------------------------------------------------------------
# Derived settings below. You usually do not need to edit anything here.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

CC12M_TENSOR_ROOT = DATA_ROOT / "cc12m_tensor_chunks"
FINETUNE_DATA_ROOT = DATA_ROOT / "finetune_wds"
MSCOCO_CAPTION_FILE = ASSET_ROOT / "coco" / "coco-30-val-2014-captions.json"
MSCOCO_STATS_FILE = ASSET_ROOT / "coco" / "mscoco_fid_stats_512.npz"

GENEVAL_METADATA = ASSET_ROOT / "geneval" / "evaluation_metadata.jsonl"
GENEVAL_MODEL_ROOT = ASSET_ROOT / "geneval" / "pretrained"
DPG_BENCH_CSV = ASSET_ROOT / "dpg_bench" / "dpg_bench.csv"
MODELSCOPE_HOME = ASSET_ROOT / "modelscope"
DPG_MPLUG_CKPT = MODELSCOPE_HOME / "iic" / "mplug_visual-question-answering_coco_large_en"

PATH_ALIASES = {
    "T2I_REPO_ROOT": REPO_ROOT,
    "T2I_DATA_ROOT": DATA_ROOT,
    "T2I_OUTPUT_ROOT": OUTPUT_ROOT,
    "T2I_CHECKPOINT_ROOT": CHECKPOINT_ROOT,
    "T2I_EVAL_OUTPUT_ROOT": EVAL_OUTPUT_ROOT,
    "T2I_ASSET_ROOT": ASSET_ROOT,
    "T2I_CC12M_TENSOR_ROOT": CC12M_TENSOR_ROOT,
    "T2I_FINETUNE_DATA_ROOT": FINETUNE_DATA_ROOT,
    "T2I_MSCOCO_CAPTION_FILE": MSCOCO_CAPTION_FILE,
    "T2I_MSCOCO_STATS_FILE": MSCOCO_STATS_FILE,
    "T2I_HF_TOKEN_FILE": HF_TOKEN_FILE,
    "T2I_WANDB_API_KEY_FILE": WANDB_API_KEY_FILE,
    "T2I_GENEVAL_METADATA": GENEVAL_METADATA,
    "T2I_GENEVAL_MODEL_ROOT": GENEVAL_MODEL_ROOT,
    "T2I_DPG_BENCH_CSV": DPG_BENCH_CSV,
    "T2I_DPG_BENCH_DATA": DPG_BENCH_DATA,
    "T2I_MODELSCOPE_HOME": MODELSCOPE_HOME,
    "T2I_DPG_MPLUG_CKPT": DPG_MPLUG_CKPT,
    "T2I_PYTHON": PYTHON,
    "T2I_GENEVAL_PYTHON": GENEVAL_PYTHON,
    "T2I_DPG_PYTHON": DPG_PYTHON,
}


def path_str(path: str | Path) -> str:
    return str(Path(path))


def expand_setting(value: Any) -> Any:
    if isinstance(value, str):
        expanded = value
        for key, path in PATH_ALIASES.items():
            expanded = expanded.replace("${" + key + "}", str(path))
        return expanded
    if isinstance(value, list):
        return [expand_setting(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_setting(item) for key, item in value.items()}
    return value
