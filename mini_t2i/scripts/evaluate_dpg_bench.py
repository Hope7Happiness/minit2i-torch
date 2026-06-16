#!/usr/bin/env python
from __future__ import annotations

import argparse
import os.path as osp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mini_t2i import settings
from mini_t2i.dpg_data import load_dpg_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation with mPLUG VQA.")
    parser.add_argument("--image-root-path", required=True)
    parser.add_argument("--data", default=str(settings.DPG_BENCH_DATA))
    parser.add_argument("--csv", dest="data", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pic-num", type=int, default=4, choices=[1, 4])
    parser.add_argument("--sample_nums", type=int, default=1065)
    parser.add_argument("--res-path", default=None)
    parser.add_argument("--vqa-model", default="mplug", choices=["mplug"])
    parser.add_argument("--mplug-ckpt", default="damo/mplug_visual-question-answering_coco_large_en")
    parser.add_argument(
        "--include-first-row",
        action="store_true",
        help="Do not skip the first CSV data row. The default mirrors the original DPG-Bench evaluator.",
    )
    return parser.parse_args()


class MPLUG(torch.nn.Module):
    def __init__(self, ckpt: str, device: str) -> None:
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image: Image.Image, question: str) -> str:
        result = self.pipeline_vqa({"image": image, "question": question})
        return result["text"]


def prepare_dpg_data(data_source: str | Path, *, skip_first_csv_row: bool = True) -> dict[str, dict]:
    question_dict: dict[str, dict] = {}
    previous_id = None
    source_path = Path(str(data_source))
    skip_first = skip_first_csv_row and source_path.exists() and source_path.suffix.lower() == ".csv"
    for row_idx, line in enumerate(load_dpg_rows(data_source)):
        if skip_first and row_idx == 0:
            continue
        current_id = line["item_id"]
        qid = int(line["proposition_id"])
        dependencies = [int(d.strip()) for d in str(line["dependency"]).split(",")]
        if current_id != previous_id:
            question_dict[current_id] = {
                "qid2tuple": {},
                "qid2dependency": {},
                "qid2question": {},
            }
        question_dict[current_id]["qid2tuple"][qid] = line["tuple"]
        question_dict[current_id]["qid2dependency"][qid] = dependencies
        question_dict[current_id]["qid2question"][qid] = line["question_natural_language"]
        previous_id = current_id
    return question_dict


def crop_image(input_image: Image.Image, crop_tuple: tuple[int, int, int, int] | None = None) -> Image.Image:
    return input_image if crop_tuple is None else input_image.crop(crop_tuple)


def compute_dpg_one_sample(
    args: argparse.Namespace,
    question_dict: dict[str, dict],
    image_path: str,
    vqa_model: MPLUG,
) -> tuple[float, dict[int, str], dict[int, float]]:
    generated_image = Image.open(image_path).convert("RGB")
    resolution = args.resolution
    crop_tuples = [
        (0, 0, resolution, resolution),
        (resolution, 0, resolution * 2, resolution),
        (0, resolution, resolution, resolution * 2),
        (resolution, resolution, resolution * 2, resolution * 2),
    ][: args.pic_num]
    key = osp.basename(image_path).rsplit(".", 1)[0]
    value = question_dict[key]
    qid2tuple = value["qid2tuple"]
    qid2question = value["qid2question"]
    qid2dependency = value["qid2dependency"]
    scores = []
    qid2scores_orig: dict[int, float] = {}
    detail_path = args.res_path.replace(".txt", "_detail.txt")
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        qid2scores: dict[int, float] = {}
        for qid, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            qid2scores[qid] = float(answer == "yes")
            with open(detail_path, "a", encoding="utf-8") as f:
                f.write(f"{image_path}, {crop_tuple}, {question}, {answer}\n")
        qid2scores_orig = qid2scores.copy()
        for qid, parent_ids in qid2dependency.items():
            if any(parent_id != 0 and qid2scores[parent_id] == 0 for parent_id in parent_ids):
                qid2scores[qid] = 0
        scores.append(sum(qid2scores.values()) / len(qid2scores))
    average_score = sum(scores) / len(scores)
    with open(args.res_path, "a", encoding="utf-8") as f:
        f.write(image_path + ", " + ", ".join(str(i) for i in scores) + ", " + str(average_score) + "\n")
    return average_score, qid2tuple, qid2scores_orig


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root_path)
    if args.res_path is None:
        args.res_path = str(image_root / f"dpg_sample{args.sample_nums}_results.txt")
    Path(args.res_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.res_path).write_text("", encoding="utf-8")
    Path(args.res_path.replace(".txt", "_detail.txt")).write_text("", encoding="utf-8")

    question_dict = prepare_dpg_data(args.data, skip_first_csv_row=not args.include_first_row)
    image_paths = sorted(p for p in image_root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    image_paths = [p for p in image_paths if p.stem in question_dict]
    device = "gpu" if torch.cuda.is_available() else "cpu"
    vqa_model = MPLUG(ckpt=args.mplug_ckpt, device=device)

    local_scores = []
    category2scores: dict[str, list[float]] = defaultdict(list)
    for n, path in enumerate(image_paths, start=1):
        try:
            score, qid2tuple, qid2scores = compute_dpg_one_sample(args, question_dict, str(path), vqa_model)
        except Exception as e:
            print(f"[dpg] failed {path}: {e}", flush=True)
            continue
        local_scores.append(score)
        for qid, tuple_text in qid2tuple.items():
            category = tuple_text.split("(")[0].strip()
            category2scores[category].append(qid2scores[qid])
        if n % 10 == 0:
            print(f"[dpg] evaluated {n}/{len(image_paths)}", flush=True)

    mean_score = float(np.mean(local_scores) * 100) if local_scores else float("nan")
    output = [f"Image path: {image_root}", f"Save results to: {args.res_path}"]
    output.append("L1 category scores:")
    l1_scores: dict[str, list[float]] = defaultdict(list)
    for category, values in category2scores.items():
        l1_scores[category.split("-")[0].strip()].extend(values)
    for category in sorted(l1_scores):
        output.append(f"\t{category}: {float(np.mean(l1_scores[category]) * 100)}")
    output.append("L2 category scores:")
    for category in sorted(category2scores):
        output.append(f"\t{category}: {float(np.mean(category2scores[category]) * 100)}")
    output.append(f"DPG-Bench score: {mean_score}")
    text = "\n".join(output)
    with open(args.res_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    simple_path = args.res_path.replace(".txt", "_simple.txt")
    Path(simple_path).write_text(str(mean_score), encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
