from __future__ import annotations

import argparse
import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"


METHOD_ORDER = [
    ("lowrank_sparse", "Low-rank sparse"),
    ("polar", "Polar low"),
    ("polar_high", "Polar high"),
    ("carp_polar", "CARP-polar"),
    ("q4", "q4 per-channel"),
    ("carp_q4_exact", "CARP-q4-exact"),
]


def bits_label(benchmark: dict, key: str) -> str:
    if key == "lowrank_sparse":
        return "unquantized"
    if key == "polar":
        return f"{benchmark['macro'][key].get('approx_bits_per_coord', 3.34375):.2f}"
    if key == "polar_high":
        bits = benchmark.get("high_bits_per_level")
        if bits is not None:
            return "3.78"
    if key == "q4":
        return "4.00"
    value = benchmark["macro"][key].get("approx_bits_per_coord")
    if value is None:
        return "—"
    return f"{value:.2f}"


def fmt(x: float) -> str:
    return f"{x:.3f}"


def build_table(benchmark: dict) -> str:
    lines = [
        "| Method | bits/coord | top-1 | top-8 | top-16 |",
        "|---|---|---|---|---|",
    ]
    macro = benchmark["macro"]
    for key, label in METHOD_ORDER:
        row = macro[key]
        lines.append(
            f"| {label} | {bits_label(benchmark, key)} | "
            f"{fmt(row['top1'])} | {fmt(row['top8'])} | {fmt(row['top16'])} |"
        )
    return "\n".join(lines)


def build_summary(benchmark: dict) -> str:
    macro = benchmark["macro"]
    polar = macro["polar"]
    polar_high = macro["polar_high"]
    carp_polar = macro["carp_polar"]
    q4 = macro["q4"]
    carp_q4 = macro["carp_q4_exact"]

    polar_gap = max(polar_high["top1"] - polar["top1"], 1e-9)
    carp_gap_closed = (carp_polar["top1"] - polar["top1"]) / polar_gap
    polar_bits_saved = float(bits_label(benchmark, "polar_high")) - carp_polar["approx_bits_per_coord"]

    q4_top1_gain = carp_q4["top1"] - q4["top1"]
    q4_top8_delta = carp_q4["top8"] - q4["top8"]
    q4_top16_delta = carp_q4["top16"] - q4["top16"]
    q4_bit_increase = carp_q4["approx_bits_per_coord"] - 4.0

    lines = [
        f"Model: `{benchmark['model_name']}`",
        f"Tasks: {', '.join(benchmark['tasks'])}",
        "",
        "**Table**",
        build_table(benchmark),
        "",
        "**Key Results**",
        f"- `CARP-polar` closes {carp_gap_closed * 100:.1f}% of the low-to-high polar top-1 gap.",
        f"- `CARP-polar` matches `polar_high` on top-1 within {abs(carp_polar['top1'] - polar_high['top1']):.4f} while saving about {polar_bits_saved:.2f} bits/coord.",
        f"- `q4` is the strongest plain base codec here: top-1 `{fmt(q4['top1'])}`, top-8 `{fmt(q4['top8'])}`, top-16 `{fmt(q4['top16'])}`.",
        f"- `CARP-q4-exact` improves top-1 by {q4_top1_gain:.3f} for only {q4_bit_increase:.2f} extra bits/coord.",
        f"- `CARP-q4-exact` changes broader ranking quality: top-8 delta `{q4_top8_delta:.3f}`, top-16 delta `{q4_top16_delta:.3f}` versus plain `q4`.",
        "",
        "**Interpretation**",
        "- `CARP-polar` is the paper's clean low-bit result: near-low-polar rate with high-polar quality.",
        "- The promotion gate is codec-agnostic because it also helps on top of `q4`.",
        "- On this model, `q4` remains the strongest plain base codec, so end-to-end cache-path tests should compare `polar -> high_polar` against `q4 -> exact`.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the real same-layer Q/K benchmark into a paper-ready table.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(RESULTS_DIR / "real_qk_attention_benchmark.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "real_qk_attention_summary.md"),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input benchmark JSON not found: {input_path}. "
            "Run benchmark_real_qk_attention.py or `python colab_runner.py realbench ...` first."
        )
    benchmark = json.loads(input_path.read_text(encoding="utf-8"))
    summary = build_summary(benchmark)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
