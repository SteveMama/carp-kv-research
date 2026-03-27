from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
LONG_BENCH_REPO = REPO_ROOT / "LongBenchRepo"
DEFAULT_TASKS = ["qasper", "multifieldqa_en", "2wikimqa"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), check=True)


def maybe_set_hf_token(token: str) -> None:
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF token set")


def ensure_longbench() -> None:
    if LONG_BENCH_REPO.exists():
        print(f"LongBench repo already present: {LONG_BENCH_REPO}")
        return
    run(["git", "clone", "https://github.com/THUDM/LongBench.git", str(LONG_BENCH_REPO)])


def cmd_setup(args: argparse.Namespace) -> None:
    maybe_set_hf_token(args.hf_token)
    print("Python:", sys.version)
    try:
        run(["nvidia-smi"])
    except Exception:
        print("nvidia-smi unavailable")

    pip_base = [sys.executable, "-m", "pip", "install", "-U"]
    if args.install_torch:
        run(
            pip_base
            + ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"]
        )
    run(
        pip_base
        + [
            "pip",
            "setuptools",
            "wheel",
            "transformers",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "datasets",
            "rank_bm25",
            "llmlingua",
            "sentence-transformers",
            "scikit-learn",
        ]
    )
    ensure_longbench()


def cmd_profile(args: argparse.Namespace) -> None:
    maybe_set_hf_token(args.hf_token)
    ensure_longbench()
    cmd = [
        sys.executable,
        "profile_qwen_kv_polar.py",
        "--model-name",
        args.model_name,
        "--tasks",
        *args.tasks,
        "--samples-per-task",
        str(args.samples_per_task),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--max-vectors-per-item",
        str(args.max_vectors_per_item),
    ]
    run(cmd)


def cmd_cache(args: argparse.Namespace) -> None:
    maybe_set_hf_token(args.hf_token)
    ensure_longbench()
    for threshold in args.exact_head_thresholds:
        cmd = [
            sys.executable,
            "carp_cache_eval.py",
            "--model-name",
            args.model_name,
            "--tasks",
            *args.tasks,
            "--samples-per-task",
            str(args.samples_per_task),
            "--full-max-context-tokens",
            str(args.full_max_context_tokens),
            "--exact-head-risk-threshold",
            str(threshold),
            "--output",
            f"results/carp_cache_eval_thr{str(threshold).replace('.', '')}_gpu.json",
        ]
        run(cmd)


def cmd_multistep(args: argparse.Namespace) -> None:
    maybe_set_hf_token(args.hf_token)
    ensure_longbench()
    cmd = [
        sys.executable,
        "carp_multistep_eval.py",
        "--model-name",
        args.model_name,
        "--tasks",
        *args.tasks,
        "--samples-per-task",
        str(args.samples_per_task),
        "--decode-steps",
        str(args.decode_steps),
        "--exact-head-risk-threshold",
        str(args.exact_head_risk_threshold),
        "--entropy-fallback-threshold",
        str(args.entropy_fallback_threshold),
        "--output",
        args.output,
    ]
    run(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab runner for CARP-KV experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Install Python dependencies and fetch LongBench.")
    setup.add_argument("--hf-token", default="")
    setup.add_argument("--install-torch", action="store_true")
    setup.set_defaults(func=cmd_setup)

    profile = sub.add_parser("profile", help="Run the offline Qwen/Llama KV profiler.")
    profile.add_argument("--hf-token", default="")
    profile.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    profile.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    profile.add_argument("--samples-per-task", type=int, default=2)
    profile.add_argument("--max-context-tokens", type=int, default=1024)
    profile.add_argument("--max-vectors-per-item", type=int, default=4096)
    profile.set_defaults(func=cmd_profile)

    cache = sub.add_parser("cache", help="Run single-step cache-path evaluation.")
    cache.add_argument("--hf-token", default="")
    cache.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    cache.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    cache.add_argument("--samples-per-task", type=int, default=5)
    cache.add_argument("--full-max-context-tokens", type=int, default=512)
    cache.add_argument("--exact-head-thresholds", nargs="+", type=float, default=[0.8, 0.7])
    cache.set_defaults(func=cmd_cache)

    multistep = sub.add_parser("multistep", help="Run multi-step CARP evaluation.")
    multistep.add_argument("--hf-token", default="")
    multistep.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    multistep.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    multistep.add_argument("--samples-per-task", type=int, default=5)
    multistep.add_argument("--decode-steps", type=int, default=8)
    multistep.add_argument("--exact-head-risk-threshold", type=float, default=0.7)
    multistep.add_argument("--entropy-fallback-threshold", type=float, default=0.30)
    multistep.add_argument(
        "--output",
        default="results/carp_multistep_eval_thr07_entropy03_gpu.json",
    )
    multistep.set_defaults(func=cmd_multistep)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
