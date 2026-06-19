<p align="center">
  <img src="assets/teaser.png" alt="MiniT2I logo" />
</p>

<h2 align="center">MiniT2I: A Minimalist Baseline for Text-to-Image Generation</h1>

<p align="center">
  <a href="https://peppaking8.github.io/#/post/minit2i"><img src="https://img.shields.io/badge/Blog-MiniT2I-2ea44f.svg" alt="MiniT2I blog post" /></a>
  &nbsp;
  <a href="https://huggingface.co/MiniT2I/MiniT2I"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-MiniT2I-yellow.svg" alt="Hugging Face checkpoints" /></a>
  &nbsp;
  <a href="https://github.com/PeppaKing8/minit2i-jax"><img src="https://img.shields.io/badge/Code-JAX-blue.svg" alt="JAX code" /></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/Hope7Happiness/minit2i-torch/blob/main/notebooks/minit2i_colab_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" /></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License" /></a>
</p>

Official PyTorch/Diffusers re-implementation of **MiniT2I**.

MiniT2I is a simple direct-RGB text-to-image generator that trains a pixel-space **MM-JiT** denoiser with flow matching, conditioned on frozen FLAN-T5-Large text tokens. The recipe is intentionally plain: avoiding image tokenizers, cascaded generation, RL stages, and any auxiliary losses. Data used in training MiniT2I is fully public and easy to implement. For more details, please refer to our [blog post](https://peppaking8.github.io/#/post/minit2i).

This repository is a PyTorch reproduction codebase for MiniT2I, including:
- PyTorch/Diffusers **inference** for MiniT2I-B/16 and MiniT2I-L/16.
- **LoRA adaptation** code used for the downstream adaptation experiments.
- Full PyTorch **training** code for the ablation setting MiniT2I-B/32.
- GenEval, DPG-Bench, and FID **evaluation** entry points.

## Table of Contents

- [Model Zoo](#model-zoo)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Diffusers Inference & Colab Demo](#diffusers-inference--colab-demo)
  - [Recommended Inference Settings](#recommended-inference-settings)
- [LoRA Adaptation](#lora-adaptation)
- [Evaluation](#evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Evaluate Checkpoints](#evaluate-checkpoints)
- [Full Training](#full-training)
  - [Dataset Preparation](#dataset-preparation)
  - [Pretrain on CC12M](#pretrain-on-cc12m)
  - [Fine-Tune on 120K mix](#fine-tune-on-120k-mix)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Model Zoo

| Model | Params | Patch | GenEval | DPG-Bench |
| --- | ---: | ---: | ---: | ---: |
| MiniT2I-B/16 | 258M + 341M text encoder | 16 | 0.873 (0.873) | 84.1 (84.2) |
| MiniT2I-L/16 | 912M + 341M text encoder | 16 | 0.878 (0.883) | 84.9 (85.9) |

The number in parentheses is the reproduction results by the [JAX implementation](https://github.com/PeppaKing8/minit2i-jax).

For checkpoints, please refer to the [Hugging Face repository](https://huggingface.co/MiniT2I/MiniT2I) -- it stores both MiniT2I-B/16 and MiniT2I-L/16 weights.

## Repository Layout

```text
.
├── main.py                     # unified train/eval entry point
├── configs/                    # pretraining, fine-tuning, and eval configs
├── mini_t2i/                    # PyTorch training and benchmark evaluation code
├── diffusers/                  # custom Diffusers pipeline source
├── lora/                       # LoRA adaptation entry points
└── tools/                      # dataset preparation helpers
```

## Installation

Create a Python environment with a CUDA-enabled PyTorch build, then install the dependencies:

```bash
git clone <REPO_URL>
cd minit2i-torch

python -m pip install -r requirements.txt
```

For inference-only use, the core dependencies are `torch`, `diffusers`, `transformers`, `safetensors`, `pillow`, and `huggingface_hub`.

Edit the user settings at the top of `mini_t2i/settings.py` before training or evaluation:

```python
DATA_ROOT = Path("/path/to/dataset/root")
OUTPUT_ROOT = Path("/path/to/output/root")
CHECKPOINT_ROOT = Path("/path/to/checkpoint/root")
EVAL_OUTPUT_ROOT = Path("/path/to/evaluation/output/root")
ASSET_ROOT = Path("/path/to/evaluation/assets")

HF_TOKEN_FILE = Path("/path/to/secrets/hf_token")
WANDB_API_KEY_FILE = Path("/path/to/secrets/wandb_api_key")
```

The rest of `mini_t2i/settings.py` derives benchmark, dataset, checkpoint, and output paths from these roots.

## Diffusers Inference & Colab Demo

Try MiniT2I without setting up training data in the [Colab demo](https://colab.research.google.com/github/Hope7Happiness/minit2i-torch/blob/main/notebooks/minit2i_colab_demo.ipynb)!

```python
import torch
from diffusers import DiffusionPipeline

HUB_MODEL_ID = "MiniT2I/MiniT2I"

pipe = DiffusionPipeline.from_pretrained(
    HUB_MODEL_ID,
    custom_pipeline=HUB_MODEL_ID,
    trust_remote_code=True,
)

image = pipe(
    "A lonely astronaut standing on a quiet beach under two moons.",
    model_type="b16",
    guidance_scale=2.5,
    num_inference_steps=100,
    torch_dtype=torch.bfloat16,
).images[0]
image.save("minit2i-b16.png")

image = pipe(
    "a transparent green backpack on a marble pedestal, with notebooks and a metal water bottle visible inside",
    model_type="l16",
    guidance_scale=6.0,
    num_inference_steps=100,
    torch_dtype=torch.bfloat16,
).images[0]
image.save("minit2i-l16.png")
```

Supported aliases include:
- `b`, `b16`, `base`, `minit2i-b/16` (for using MiniT2I-B/16);
- `l`, `l16`, `large`, and `minit2i-l/16` (for using MiniT2I-L/16).

### Recommended Inference Settings

The following values are the defaults used by the Colab demo and the GenEval / DPG-Bench evaluation configs. They are reasonable starting points; CFG can be tuned per prompt.

| Setting | MiniT2I-B/16 | MiniT2I-L/16 | Source |
| --- | --- | --- | --- |
| `num_inference_steps` | 100 | 100 | Colab demo & training `n_T` |
| `guidance_scale` (general use) | 2.5 | 6.0 | Colab demo |
| `guidance_scale` (GenEval / DPG-Bench) | 5.0 | 5.0 | `configs/eval_hf_*.yml` |
| `torch_dtype` | `bfloat16` | `bfloat16` | Colab demo |
| Resolution | 512 × 512 | 512 × 512 | Training image size |

## LoRA Adaptation

The LoRA code attaches adapters to the attention projections, MLP projections, and text projection layers. 

Here, we provide our default settings in the blog post: using rank-32 adapters on the Naruto BLIP-caption dataset as a domain-adaptation example.

```bash
python lora/train_lora.py \
  --pretrained_model_name_or_path MiniT2I/MiniT2I \
  --model_type b16 \
  --dataset_name lambdalabs/naruto-blip-captions \
  --output_dir lora_runs/naruto_lora \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --rank 32 \
  --lora_alpha 32 \
  --mixed_precision bf16 \
  --loss_type velocity \
  --max_train_steps 400 \
  --validation_at_start \
  --validation_steps_interval 100 \
  --validation_prompt "anime portrait of a ninja with orange hair, high quality"
```

To sample a trained adapter:

```bash
python lora/sample_lora.py \
  --adapter_dir lora_runs/naruto_lora \
  --prompt "anime portrait of a ninja with orange hair, high quality" \
  --steps 100 \
  --cfg_scale 6.0 \
  --device cuda \
  --dtype bfloat16
```

For an end-to-end walkthrough of the same recipe, see the [`notebooks/minit2i_lora_naruto_finetune.ipynb`](notebooks/minit2i_lora_naruto_finetune.ipynb) notebook.

## Evaluation

The evaluation pipeline supports GenEval and DPG-Bench prompt-following evaluation for the Hugging Face Diffusers checkpoints and local PyTorch checkpoints.

### Evaluation Metrics

This repository does not vendor benchmark prompts, detector checkpoints, or VQA checkpoints. Download each benchmark asset separately and edit `mini_t2i/settings.py`.

**GenEval.** GenEval uses prompts and detector checkpoints from [`djghosh13/geneval`](https://github.com/djghosh13/geneval). Follow that repository to download the prompt metadata and pretrained detector assets. With the default settings, place them as:

```text
/path/to/evaluation/assets/geneval/evaluation_metadata.jsonl
/path/to/evaluation/assets/geneval/pretrained/
```

**DPG-Bench.** DPG-Bench uses the [`Jialuo21/DPG-Bench`](https://huggingface.co/datasets/Jialuo21/DPG-Bench) Hugging Face dataset and the mPLUG VQA checkpoint [`iic/mplug_visual-question-answering_coco_large_en`](https://www.modelscope.cn/models/iic/mplug_visual-question-answering_coco_large_en/summary). The Hugging Face dataset is stored as parquet with a `test` split, so the default config loads it directly with `datasets` instead of requiring a CSV download.

With the default settings, place the mPLUG checkpoint as:

```text
/path/to/evaluation/assets/modelscope/iic/mplug_visual-question-answering_coco_large_en/
```

To use a local DPG-Bench file instead, set `DPG_BENCH_DATA` in `mini_t2i/settings.py` to a local CSV, parquet, JSON, or JSONL path.

### Evaluate Checkpoints

Evaluate the Hugging Face Diffusers checkpoints directly from the Hub without downloading a local `.pt` file:

```bash
python main.py eval --config-file configs/eval_hf_b16.yml
python main.py eval --config-file configs/eval_hf_l16.yml
```

These configs use `hf_model_id: MiniT2I/MiniT2I` and `model_type: b16` or `model_type: l16`. Edit `configs/eval.yml` only when evaluating a local PyTorch checkpoint:

```yaml
checkpoint: /path/to/checkpoint.pt
checkpoint_target: ema
output_dir: /path/to/eval_output
```

Run all enabled benchmarks:

```bash
python main.py eval --config-file configs/eval.yml
```

## Full Training

The PyTorch training interface is config-driven through `main.py`, `configs/`, and `mini_t2i/`.

We provide our default B/32 ablation settings in `configs/pretrain.yml` and `configs/finetune.yml`. For the training config of B/16 and L/16, please refer to our [JAX code release](https://github.com/PeppaKing8/minit2i-jax).

### Dataset Preparation

This repository does not vendor training datasets. Download the data separately, prepare it into the expected local layout, and edit `mini_t2i/settings.py` to point at your copies.

**CC12M.** Pretraining uses Conceptual Captions 12M data. Prepare local tensor chunks from [`CaptionEmporium/conceptual-captions-cc12m-llavanext`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext):

```bash
python tools/prepare_cc12m_chunks.py \
  --dataset CaptionEmporium/conceptual-captions-cc12m-llavanext \
  --out /path/to/dataset/root/cc12m_tensor_chunks
```

With the default settings, the chunks should live at `/path/to/dataset/root/cc12m_tensor_chunks`.

**120K mix.** Fine-tuning uses the 120K mix WebDataset layout assembled from three public shards:

- [`BLIP3o/BLIP3o-60k`](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)
- [`CaptionEmporium/dalle3-llama3.2-11b`](https://huggingface.co/datasets/CaptionEmporium/dalle3-llama3.2-11b)
- [`FreedomIntelligence/ShareGPT-4o-Image`](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)

Stage the shards into the directory structure expected by `configs/finetune.yml`:

```bash
python tools/prepare_120k_mix_layout.py \
  --out /path/to/dataset/root/finetune_wds
```

- With the default settings, the WebDataset shards should live at `/path/to/dataset/root/finetune_wds`.
- The fine-tuning loader samples the three source groups with weights `[0.06, 0.016, 0.04]` for BLIP3o-60K, DALL-E3, and ShareGPT4o, _roughly proportional to the number of samples_. 
- When local shards are absent and `finetune_hf_streaming: true`, the loader falls back to the public Hugging Face sources configured in `TrainConfig`; _we recommend staging local WebDataset shards_ for speed.

### Pretrain on CC12M

```bash
torchrun --standalone --nproc_per_node=8 \
  main.py train \
  --config-file configs/pretrain.yml \
  --output-dir "${OUT_DIR}" \
  --dataset-backend local_folder \
  --micro-batch-size 128 \
  --grad-accum-steps 1 \
  --attention-impl sdpa
```

MSCOCO-30K FID is the online monitoring metric used during pretraining. Download the caption file and reference statistics from [`MiniT2I/evaluation-assets`](https://huggingface.co/datasets/MiniT2I/evaluation-assets/tree/main/coco):

```bash
huggingface-cli download MiniT2I/evaluation-assets \
  --repo-type dataset \
  --include "coco/*" \
  --local-dir /path/to/evaluation/assets
```

With the default settings, this command creates the expected `coco/` asset directory. The repository FID implementation uses `pytorch-fid`'s InceptionV3 feature extractor and compares generated-image features against `mscoco_fid_stats_512.npz`.

Pretraining enables online FID monitoring by default. `configs/pretrain.yml` sets:

```yaml
fid_every: 20000
fid_num_samples: 30000
fid_batch_size: 64
```

The training loop evaluates both the raw model and the EMA model, then logs the FID metrics to [W&B](wandb.ai) when enabled. Edit `MSCOCO_CAPTION_FILE` and `MSCOCO_STATS_FILE` in `mini_t2i/settings.py` before launching pretraining.

To run the same FID path manually for a locally trained pretrained checkpoint, enable the `fid` benchmark in `configs/eval.yml`:

```yaml
checkpoint: /path/to/pretrained/checkpoint.pt
checkpoint_target: ema
output_dir: /path/to/eval_output

benchmarks:
  fid:
    enabled: true
    num_samples: 30000
    batch_size: 64
```

### Fine-Tune on 120K mix

You can directly run the following script to reproduce our B/32 ablation-level results. Hyperparameters can be tuned at `configs/finetune.yml`.

```bash
torchrun --standalone --nproc_per_node=8 \
  main.py train \
  --config-file configs/finetune.yml \
  --resume-from "${PRETRAINED_CKPT}" \
  --no-auto-resume \
  --output-dir "${OUT_DIR}" \
  --micro-batch-size 64 \
  --grad-accum-steps 2 \
  --num-workers 8 \
  --attention-impl sdpa
```

## Acknowledgments

This codebase builds on a number of open-source efforts:

- [JAX MiniT2I](https://github.com/PeppaKing8/minit2i-jax) -- the reference implementation that this PyTorch port mirrors.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and [Transformers](https://github.com/huggingface/transformers) -- pipeline scaffolding and the FLAN-T5-Large text encoder.
- [PEFT](https://github.com/huggingface/peft) -- LoRA adapters used in `lora/`.
- [`djghosh13/geneval`](https://github.com/djghosh13/geneval) -- prompts and detector checkpoints for GenEval.
- [`Jialuo21/DPG-Bench`](https://huggingface.co/datasets/Jialuo21/DPG-Bench) and the mPLUG VQA model from ModelScope -- DPG-Bench evaluation.
- [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid) -- InceptionV3 FID feature extraction.
- Public training data from [`CaptionEmporium/conceptual-captions-cc12m-llavanext`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext), [`BLIP3o/BLIP3o-60k`](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k), [`CaptionEmporium/dalle3-llama3.2-11b`](https://huggingface.co/datasets/CaptionEmporium/dalle3-llama3.2-11b), [`FreedomIntelligence/ShareGPT-4o-Image`](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image), and [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions).

## Citation

```bibtex
@misc{minit2i2026,
  title  = {MiniT2I: A Minimalist Baseline for Text-to-Image Generation},
  author = {Wang, Xianbang and Zhao, Hanhong and Lu, Yiyang and Zhou, Kangyang and Ma, Linrui and He, Kaiming},
  year   = {2026},
  url    = {https://peppaking8.github.io/#/post/minit2i}
}
```
