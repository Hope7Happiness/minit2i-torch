from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DATASETS = {
    "blip3_ft60k": "BLIP3o/BLIP3o-60k",
    "dalle3": "CaptionEmporium/dalle3-llama3.2-11b",
    "sharegpt4o": "FreedomIntelligence/ShareGPT-4o-Image",
}

EXPECTED = {
    "blip3_ft60k": [
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
    ],
    "dalle3": ["shard-*.tar"],
    "sharegpt4o": ["shard-*.tar", "text_to_image_part_*.tar"],
}


def link_or_copy(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    parser = argparse.ArgumentParser(description="Stage 120K mix WebDataset tar files in the layout expected by configs/finetune.yml.")
    parser.add_argument("--out", required=True, help="Destination directory used as FINETUNE_DATA_ROOT in mini_t2i/settings.py.")
    parser.add_argument("--cache-dir", default="hf_cache")
    parser.add_argument("--copy", action="store_true", help="Copy tar files instead of symlinking them from the Hugging Face cache.")
    parser.add_argument("--source", choices=sorted(DATASETS), action="append", default=None)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out_root = Path(args.out)
    sources = args.source or list(DATASETS)
    for name in sources:
        repo_id = DATASETS[name]
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=args.cache_dir,
                allow_patterns=["*.tar", "**/*.tar"],
            )
        )
        tar_files = sorted(snapshot.rglob("*.tar"))
        if not tar_files:
            raise RuntimeError(f"no .tar files found in {repo_id}")
        dst_dir = out_root / name
        for src in tar_files:
            link_or_copy(src, dst_dir / src.name, args.copy)
        print(f"staged {len(tar_files)} tar files for {name} -> {dst_dir}", flush=True)

        missing = []
        for pattern in EXPECTED[name]:
            if not list(dst_dir.glob(pattern)):
                missing.append(pattern)
        if missing:
            print(f"warning: {name} is missing expected pattern(s): {', '.join(missing)}", flush=True)


if __name__ == "__main__":
    main()
