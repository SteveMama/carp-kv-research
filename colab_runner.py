from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
from urllib.request import urlretrieve
import zipfile


REPO_ROOT = Path(__file__).resolve().parent
LONG_BENCH_REPO = REPO_ROOT / "LongBenchRepo"
VENV_DIR = REPO_ROOT / ".venv-colab"
DEFAULT_TASKS = ["qasper", "multifieldqa_en", "2wikimqa"]
LONGBENCH_DATA_ZIP_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), check=True)


def maybe_set_hf_token(token: str) -> None:
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF token set")


def resolve_venv_python() -> Path | None:
    candidates = [
        VENV_DIR / "bin" / "python",
        VENV_DIR / "bin" / "python3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_longbench_base(repo_root: Path) -> Path:
    candidates = [
        repo_root / "LongBench",
        repo_root,
    ]
    for candidate in candidates:
        if (candidate / "config").exists():
            return candidate
    return repo_root / "LongBench" if (repo_root / "LongBench").exists() else repo_root


def resolve_longbench_bases(repo_root: Path) -> list[Path]:
    bases: list[Path] = []
    for candidate in [repo_root / "LongBench", repo_root]:
        if (candidate / "config").exists() and candidate not in bases:
            bases.append(candidate)
    if not bases:
        fallback = resolve_longbench_base(repo_root)
        bases.append(fallback)
    return bases


def longbench_layout_valid(repo_root: Path) -> bool:
    for base in resolve_longbench_bases(repo_root):
        if (base / "config").exists() and (base / "data").exists() and any((base / "data").glob("*.jsonl")):
            return True
    return False


def download_longbench_data(repo_root: Path) -> None:
    bases = resolve_longbench_bases(repo_root)
    if not any((base / "config").exists() for base in bases):
        raise RuntimeError(f"LongBench config directory missing under {repo_root}")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "data.zip"
        unpack_dir = Path(tmpdir) / "unzipped"
        print(f"Downloading LongBench data archive from {LONGBENCH_DATA_ZIP_URL}")
        urlretrieve(LONGBENCH_DATA_ZIP_URL, archive)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(unpack_dir)
        extracted_data = unpack_dir / "data"
        if not extracted_data.exists():
            nested = next((path for path in unpack_dir.rglob("data") if path.is_dir()), None)
            if nested is None:
                raise RuntimeError(f"Downloaded LongBench archive but could not locate a data directory under {unpack_dir}")
            extracted_data = nested
        jsonl_count = sum(1 for _ in extracted_data.glob("*.jsonl"))
        if jsonl_count == 0:
            raise RuntimeError(f"Downloaded LongBench archive but found no *.jsonl files under {extracted_data}")
        for base in bases:
            if not (base / "config").exists():
                continue
            target = base / "data"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extracted_data, target, dirs_exist_ok=True)
            print(f"Installed LongBench data into {target} ({jsonl_count} files)")


def ensure_longbench() -> None:
    if LONG_BENCH_REPO.exists() and longbench_layout_valid(LONG_BENCH_REPO):
        print(f"LongBench repo already present: {LONG_BENCH_REPO}")
        return
    if LONG_BENCH_REPO.exists():
        print(f"LongBench repo exists but is invalid, recloning: {LONG_BENCH_REPO}")
        shutil.rmtree(LONG_BENCH_REPO)
    run(["git", "clone", "https://github.com/THUDM/LongBench.git", str(LONG_BENCH_REPO)])
    if not longbench_layout_valid(LONG_BENCH_REPO):
        download_longbench_data(LONG_BENCH_REPO)
    if not longbench_layout_valid(LONG_BENCH_REPO):
        raise RuntimeError(
            f"Cloned LongBench repo but still could not find a valid data/config layout under {LONG_BENCH_REPO}"
        )


def maybe_reexec_into_venv(command: str) -> None:
    if command == "setup":
        return
    venv_python = resolve_venv_python()
    if venv_python is None:
        return
    current = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if current == target:
        return
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


def cmd_setup(args: argparse.Namespace) -> None:
    maybe_set_hf_token(args.hf_token)
    print("Python:", sys.version)
    try:
        run(["nvidia-smi"])
    except Exception:
        print("nvidia-smi unavailable")

    venv_python = resolve_venv_python()
    if venv_python is None:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        venv_python = resolve_venv_python()
    if venv_python is None:
        raise RuntimeError(f"Failed to create a usable virtual environment under {VENV_DIR}")

    pip_base = [str(venv_python), "-m", "pip", "install", "-U"]
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
    run([str(venv_python), "-V"])
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
            "--base-codec",
            args.base_codec,
            "--upgrade-codec",
            args.upgrade_codec,
            "--selector-mode",
            args.selector_mode,
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
        "--base-codec",
        args.base_codec,
        "--upgrade-codec",
        args.upgrade_codec,
        "--selector-mode",
        args.selector_mode,
        "--exact-head-risk-threshold",
        str(args.exact_head_risk_threshold),
        "--entropy-fallback-threshold",
        str(args.entropy_fallback_threshold),
        "--output",
        args.output,
    ]
    run(cmd)


def cmd_diagnose(args: argparse.Namespace) -> None:
    maybe_set_hf_token(args.hf_token)
    ensure_longbench()
    cmd = [
        sys.executable,
        "diagnose_qwen_kv_protocol.py",
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
        "--eval-keys",
        str(args.eval_keys),
        "--eval-queries",
        str(args.eval_queries),
        "--output",
        args.output,
    ]
    run(cmd)


def cmd_realbench(args: argparse.Namespace) -> None:
    maybe_set_hf_token(args.hf_token)
    ensure_longbench()
    cmd = [
        sys.executable,
        "benchmark_real_qk_attention.py",
        "--model-name",
        args.model_name,
        "--tasks",
        *args.tasks,
        "--samples-per-task",
        str(args.samples_per_task),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--min-query-pos",
        str(args.min_query_pos),
        "--output",
        args.output,
    ]
    run(cmd)


def cmd_summarize_realbench(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "summarize_real_qk_benchmark.py",
        "--input",
        args.input,
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

    profile = sub.add_parser("profile", help="Run the legacy mixed-layer key-only proxy profiler.")
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
    cache.add_argument("--base-codec", choices=["polar", "q4"], default="polar")
    cache.add_argument("--upgrade-codec", choices=["high_polar", "exact"], default="high_polar")
    cache.add_argument("--selector-mode", choices=["learned", "heuristic"], default="learned")
    cache.add_argument("--exact-head-thresholds", nargs="+", type=float, default=[0.8, 0.7])
    cache.set_defaults(func=cmd_cache)

    multistep = sub.add_parser("multistep", help="Run multi-step CARP evaluation.")
    multistep.add_argument("--hf-token", default="")
    multistep.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    multistep.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    multistep.add_argument("--samples-per-task", type=int, default=5)
    multistep.add_argument("--decode-steps", type=int, default=8)
    multistep.add_argument("--base-codec", choices=["polar", "q4"], default="polar")
    multistep.add_argument("--upgrade-codec", choices=["high_polar", "exact"], default="high_polar")
    multistep.add_argument("--selector-mode", choices=["learned", "heuristic"], default="learned")
    multistep.add_argument("--exact-head-risk-threshold", type=float, default=0.7)
    multistep.add_argument("--entropy-fallback-threshold", type=float, default=0.30)
    multistep.add_argument(
        "--output",
        default="results/carp_multistep_eval_thr07_entropy03_gpu.json",
    )
    multistep.set_defaults(func=cmd_multistep)

    diagnose = sub.add_parser("diagnose", help="Run legacy same-item vs cross-item proxy diagnosis.")
    diagnose.add_argument("--hf-token", default="")
    diagnose.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    diagnose.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    diagnose.add_argument("--samples-per-task", type=int, default=1)
    diagnose.add_argument("--max-context-tokens", type=int, default=1024)
    diagnose.add_argument("--max-vectors-per-item", type=int, default=4096)
    diagnose.add_argument("--eval-keys", type=int, default=2048)
    diagnose.add_argument("--eval-queries", type=int, default=256)
    diagnose.add_argument(
        "--output",
        default="results/qwen_protocol_diagnosis.json",
    )
    diagnose.set_defaults(func=cmd_diagnose)

    realbench = sub.add_parser("realbench", help="Run the primary real same-layer Q/K attention benchmark.")
    realbench.add_argument("--hf-token", default="")
    realbench.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    realbench.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    realbench.add_argument("--samples-per-task", type=int, default=1)
    realbench.add_argument("--max-context-tokens", type=int, default=1024)
    realbench.add_argument("--min-query-pos", type=int, default=32)
    realbench.add_argument(
        "--output",
        default="results/real_qk_attention_benchmark.json",
    )
    realbench.set_defaults(func=cmd_realbench)

    summarize = sub.add_parser("summarize-realbench", help="Summarize the real Q/K benchmark into a paper-ready table.")
    summarize.add_argument("--input", default="results/real_qk_attention_benchmark.json")
    summarize.add_argument("--output", default="results/real_qk_attention_summary.md")
    summarize.set_defaults(func=cmd_summarize_realbench)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    maybe_reexec_into_venv(args.command)
    args.func(args)


if __name__ == "__main__":
    main()
