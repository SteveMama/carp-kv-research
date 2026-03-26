# Baseline Comparison Note

This note compares:

1. the user-provided LongBench-V1 table on `Llama-3.1-8B-Instruct`
2. the current Qwen-based synthetic conversation benchmark in this repo

These are **not** directly comparable benchmarks. The comparison below is only meant to normalize them onto a relative-retention view.

## LongBench-V1 Relative Retention

Using the user-provided `Exact = 45.71` baseline:

| Method | Average | Retention vs Exact |
|---|---:|---:|
| SnapKV | 38.23 | 83.6% |
| HeadKV | 39.45 | 86.3% |
| PyramidKV | 36.80 | 80.5% |
| StreamingLLM | 25.68 | 56.2% |
| KIVI | 43.38 | 94.9% |
| PolarQuant | 44.03 | 96.3% |
| PolarQuant-R offline | 44.71 | 97.8% |
| PolarQuant-R online | 45.45 | 99.4% |

## Our Current Qwen Comparison

Using the current 3-seed robustness summary:

| Method | Exact match mean | Relative to full-context Qwen |
|---|---:|---:|
| `causal_full` | 0.4722 | 100.0% |
| `qagraph_topk` | 0.8611 | 182.4% |

Context usage:

| Method | Avg context tokens |
|---|---:|
| `causal_full` | 758.4 |
| `qagraph_topk` | 26.2 |

This is a `28.9x` context reduction.

## Caveat

The retention numbers are not apples-to-apples:

- LongBench-V1 numbers are on `Llama-3.1-8B-Instruct`
- our numbers are on `Qwen/Qwen2.5-0.5B-Instruct`
- LongBench measures standard long-context downstream performance
- our benchmark measures conversational retrieval under corrections, aliases, and multi-hop references
- our framework changes the memory-selection problem itself, rather than only compressing the KV cache

So the defensible claim is:

- on our synthetic conversational benchmark, the framework materially outperforms the Mac-runnable baselines we tested
- the LongBench table shows that strong KV codecs like `PolarQuant-R` retain almost all performance on Llama-8B
- until we run our framework on a public benchmark, we should not claim superiority over those published methods

