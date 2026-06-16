from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from mini_t2i import settings

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "mini_t2i"
DEFAULT_PYTHON = settings.PYTHON
DEFAULT_GENEVAL_PYTHON = settings.GENEVAL_PYTHON
DEFAULT_DPG_PYTHON = settings.DPG_PYTHON


def load_eval_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"eval config must be a YAML mapping: {path}")
    return settings.expand_setting(cfg)


def env_with_repo(include_runtime_deps: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    runtime_deps = str(ROOT / "runtime_deps" / "pillow_webp")
    runtime_deps_rel = str(Path("runtime_deps") / "pillow_webp")
    parts = [str(ROOT)]
    if include_runtime_deps:
        parts.insert(0, runtime_deps)
    if existing:
        parts.extend(
            part
            for part in existing.split(":")
            if part
            and part != runtime_deps
            and part != runtime_deps_rel
            and Path(part).resolve() != Path(runtime_deps).resolve()
        )
    env["PYTHONPATH"] = ":".join(parts)
    return env


def run_command(cmd: list[str], *, cwd: Path = ROOT, dry_run: bool = False, env: dict[str, str] | None = None) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[eval] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def maybe_torchrun(nproc: int, args: list[str]) -> list[str]:
    if nproc <= 1:
        return args
    script_args = args
    if args and Path(args[0]).name.startswith("python"):
        script_args = args[1:]
    return ["torchrun", "--standalone", f"--nproc_per_node={nproc}", *script_args]


def run_fid(cfg: dict[str, Any], bench: dict[str, Any], dry_run: bool) -> None:
    if cfg.get("hf_model_id"):
        raise ValueError("FID eval currently requires a local training checkpoint; disable fid for hf_model_id eval configs.")
    python = cfg.get("python", DEFAULT_PYTHON)
    cmd = [
        python,
        "-m",
        "mini_t2i.train",
        "--config-file",
        str(cfg["train_config"]),
        "--resume-from",
        str(cfg["checkpoint"]),
        "--no-auto-resume",
        "--eval-only",
        "--eval-target",
        cfg.get("checkpoint_target", "ema"),
        "--output-dir",
        str(Path(cfg["output_dir"]) / "fid"),
        "--no-wandb",
    ]
    if bench.get("num_samples") is not None:
        cmd.extend(["--fid-num-samples", str(bench["num_samples"])])
    if bench.get("batch_size") is not None:
        cmd.extend(["--fid-batch-size", str(bench["batch_size"])])
    run_command(maybe_torchrun(int(bench.get("nproc", cfg.get("nproc", 1))), cmd), dry_run=dry_run, env=env_with_repo())


def run_generation(cfg: dict[str, Any], benchmark: str, bench: dict[str, Any], outdir: Path, dry_run: bool) -> None:
    python = cfg.get("python", DEFAULT_PYTHON)
    cmd = [
        python,
        str(PACKAGE_ROOT / "scripts" / "generate_benchmark_images.py"),
        "--benchmark",
        benchmark,
        "--config-file",
        str(cfg["train_config"]),
        "--checkpoint-target",
        cfg.get("checkpoint_target", "ema"),
        "--outdir",
        str(outdir),
        "--samples-per-prompt",
        str(bench.get("samples_per_prompt", 4)),
        "--batch-size",
        str(bench.get("batch_size", 4)),
        "--attention-impl",
        str(bench.get("attention_impl", "sdpa")),
    ]
    if cfg.get("hf_model_id"):
        cmd.extend(["--hf-model-id", str(cfg["hf_model_id"])])
        cmd.extend(["--model-type", str(cfg.get("model_type", bench.get("model_type", "b16")))])
        if cfg.get("cache_dir"):
            cmd.extend(["--cache-dir", str(cfg["cache_dir"])])
    else:
        cmd.extend(["--checkpoint", str(cfg["checkpoint"])])
    if bench.get("limit") is not None:
        cmd.extend(["--limit", str(bench["limit"])])
    if bench.get("steps") is not None:
        cmd.extend(["--steps", str(bench["steps"])])
    if bench.get("cfg_scale") is not None:
        cmd.extend(["--cfg-scale", str(bench["cfg_scale"])])
    if benchmark == "geneval" and bench.get("metadata") is not None:
        cmd.extend(["--geneval-metadata", str(bench["metadata"])])
    if benchmark == "dpg":
        dpg_data = bench.get("data", bench.get("csv"))
        if dpg_data is not None:
            cmd.extend(["--dpg-data", str(dpg_data)])
    run_command(maybe_torchrun(int(bench.get("nproc", cfg.get("nproc", 1))), cmd), dry_run=dry_run, env=env_with_repo())


def run_geneval_eval(cfg: dict[str, Any], bench: dict[str, Any], outdir: Path, dry_run: bool) -> None:
    python = cfg.get("geneval_python", DEFAULT_GENEVAL_PYTHON)
    result_dir = Path(cfg["output_dir"]) / "geneval_results"
    results_jsonl = result_dir / "results.jsonl"
    summary_json = result_dir / "summary.json"
    if not dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
    if summary_json.exists():
        print(f"[eval] skip existing GenEval summary: {summary_json}", flush=True)
        return
    model_path = bench.get("model_path", str(settings.GENEVAL_MODEL_ROOT))
    eval_script = PACKAGE_ROOT / "evaluation" / "geneval" / "evaluate_images.py"
    summary_script = PACKAGE_ROOT / "evaluation" / "geneval" / "summary_scores.py"
    run_command(
        [
            python,
            str(eval_script),
            str(outdir),
            "--outfile",
            str(results_jsonl),
            "--model-path",
            str(model_path),
        ],
        cwd=ROOT,
        dry_run=dry_run,
        env=env_with_repo(include_runtime_deps=False),
    )
    run_command(
        [python, str(summary_script), str(results_jsonl), str(summary_json)],
        cwd=ROOT,
        dry_run=dry_run,
        env=env_with_repo(include_runtime_deps=False),
    )


def run_dpg_eval(cfg: dict[str, Any], bench: dict[str, Any], outdir: Path, dry_run: bool) -> None:
    python = cfg.get("dpg_python", DEFAULT_DPG_PYTHON)
    result_dir = Path(cfg["output_dir"]) / "dpg_results"
    if not dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
    if (result_dir / "dpg_results_simple.txt").exists():
        print(f"[eval] skip existing DPG result: {result_dir / 'dpg_results_simple.txt'}", flush=True)
        return
    dpg_env = env_with_repo(include_runtime_deps=False)
    modelscope_home = Path(bench.get("modelscope_home", settings.MODELSCOPE_HOME))
    if not dry_run:
        modelscope_home.mkdir(parents=True, exist_ok=True)
    dpg_env.setdefault("MODELSCOPE_CACHE", str(modelscope_home))
    dpg_env.setdefault("MODELSCOPE_HOME", str(modelscope_home))
    dpg_env.setdefault("XDG_CACHE_HOME", str(Path(cfg["output_dir"]) / "cache"))
    dpg_env.setdefault("HOME", str(ROOT))
    cmd = [
        python,
        str(PACKAGE_ROOT / "scripts" / "evaluate_dpg_bench.py"),
        "--image-root-path",
        str(outdir),
        "--data",
        str(bench.get("data", bench.get("csv", settings.DPG_BENCH_DATA))),
        "--resolution",
        str(bench.get("resolution", 512)),
        "--pic-num",
        str(bench.get("samples_per_prompt", 4)),
        "--res-path",
        str(result_dir / "dpg_results.txt"),
    ]
    if bench.get("sample_nums") is not None:
        cmd.extend(["--sample_nums", str(bench["sample_nums"])])
    if bench.get("mplug_ckpt") is not None:
        cmd.extend(["--mplug-ckpt", str(bench["mplug_ckpt"])])
    run_command(cmd, dry_run=dry_run, env=dpg_env)


def run_eval_config(cfg: dict[str, Any], dry_run: bool = False) -> None:
    required = ("train_config", "output_dir")
    for key in required:
        if not cfg.get(key):
            raise KeyError(f"eval config missing required key: {key}")
    if not cfg.get("checkpoint") and not cfg.get("hf_model_id"):
        raise KeyError("eval config requires either checkpoint or hf_model_id")
    if not dry_run:
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    benchmarks = cfg.get("benchmarks", {})
    if benchmarks.get("fid", {}).get("enabled", False):
        run_fid(cfg, benchmarks["fid"], dry_run)
    if benchmarks.get("geneval", {}).get("enabled", False):
        bench = benchmarks["geneval"]
        outdir = Path(bench.get("outdir", Path(cfg["output_dir"]) / "geneval_images"))
        summary_json = Path(cfg["output_dir"]) / "geneval_results" / "summary.json"
        if summary_json.exists():
            print(f"[eval] skip existing GenEval result: {summary_json}", flush=True)
        else:
            if bench.get("generate", True):
                run_generation(cfg, "geneval", bench, outdir, dry_run)
            if bench.get("evaluate", True):
                run_geneval_eval(cfg, bench, outdir, dry_run)
    if benchmarks.get("dpg", {}).get("enabled", False):
        bench = benchmarks["dpg"]
        outdir = Path(bench.get("outdir", Path(cfg["output_dir"]) / "dpg_images"))
        simple_result = Path(cfg["output_dir"]) / "dpg_results" / "dpg_results_simple.txt"
        if simple_result.exists():
            print(f"[eval] skip existing DPG result: {simple_result}", flush=True)
        else:
            if bench.get("generate", True):
                run_generation(cfg, "dpg", bench, outdir, dry_run)
            if bench.get("evaluate", True):
                run_dpg_eval(cfg, bench, outdir, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured mini_t2i evaluations.")
    parser.add_argument("--config-file", default=str(ROOT / "configs" / "eval.yml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_eval_config(load_eval_config(args.config_file), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
