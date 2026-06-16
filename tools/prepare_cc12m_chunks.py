from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def first_existing(row: dict, candidates: list[str]) -> str:
    for name in candidates:
        if name in row and row[name] is not None:
            return name
    raise KeyError(f"none of these columns were found: {', '.join(candidates)}")


def to_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "path" in value:
        return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        request = urllib.request.Request(value, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    return Image.open(value).convert("RGB")


def write_chunk(rows, out_dir: Path, chunk_idx: int, tokenizer, transform, image_column: str, caption_column: str, prompt_length: int):
    images = []
    captions = []
    for row in rows:
        image = to_image(row[image_column])
        images.append(transform(image))
        captions.append(str(row[caption_column]))

    tokens = tokenizer(
        captions,
        max_length=prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    payload = {
        "pixel_values": torch.stack(images, dim=0),
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "caption": captions,
    }
    path = out_dir / f"chunk_{chunk_idx:06d}.pt"
    torch.save(payload, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Prepare MiniT2I pretraining tensor chunks from a Hugging Face image/text dataset.")
    parser.add_argument("--dataset", default="CaptionEmporium/conceptual-captions-cc12m-llavanext")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-column", default="")
    parser.add_argument("--caption-column", default="")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokenizer", default="google/flan-t5-large")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=args.prompt_length)
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.PILToTensor(),
        ]
    )

    image_column = args.image_column
    caption_column = args.caption_column
    image_candidates = ["image", "jpg", "jpeg", "png", "url"]
    caption_candidates = ["caption_llava", "caption_llava_short", "text", "caption", "llava_caption", "recaption", "re_caption", "prompt"]

    rows = []
    chunk_idx = 0
    total = 0
    for row in dataset:
        if not image_column:
            image_column = first_existing(row, image_candidates)
        if not caption_column:
            caption_column = first_existing(row, caption_candidates)
        rows.append(row)
        total += 1
        if len(rows) == args.chunk_size:
            path = write_chunk(rows, out_dir, chunk_idx, tokenizer, transform, image_column, caption_column, args.prompt_length)
            print(f"wrote {path} ({total} samples)", flush=True)
            rows.clear()
            chunk_idx += 1
        if args.limit and total >= args.limit:
            break

    if rows:
        path = write_chunk(rows, out_dir, chunk_idx, tokenizer, transform, image_column, caption_column, args.prompt_length)
        print(f"wrote {path} ({total} samples)", flush=True)


if __name__ == "__main__":
    main()
