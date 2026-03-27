# CARP-KV Diagnostics

This repository started as a KV-cache compression project and ended up producing a stronger result in a different direction:

- a corrected evaluation protocol for KV-cache codecs
- a failure taxonomy for token-level, head-level, and decode-step errors
- a small mixed-precision baseline that is useful in practice on `Qwen/Qwen2.5-0.5B-Instruct`

The current honest framing is:

> This repo is primarily a diagnostic and evaluation framework for KV-cache quantization, with `CARP` as the experimental mixed-precision policy built on top of that framework.

## Current Conclusions

### 1. The old proxy benchmark was wrong

The early `profile_qwen_kv_polar.py` benchmark:

- used held-out keys as pseudo-queries
- mixed keys across layers

That proxy made codec quality look dramatically different from reality. Under the corrected benchmark, polar quantization recovered from `~0.10` top-1 to `~0.81` top-1 on real attention.

### 2. Real same-layer Q/K benchmarking is the correct offline test

The canonical offline benchmark in this repo is now:

- real post-RoPE queries
- real post-RoPE keys
- same layer
- same KV head
- causal key prefix per query position

Script:

- [`benchmark_real_qk_attention.py`](./benchmark_real_qk_attention.py)

### 3. `CARP-polar` is a real result under the corrected benchmark

On `Qwen/Qwen2.5-0.5B-Instruct`, with 138,240 real query/key evaluations:

| Method | Bits/coord | Top-1 | Top-8 | Top-16 |
|---|---:|---:|---:|---:|
| Polar low `(4,3,2,2,2,2)` | 3.34 | 0.809 | 0.954 | 0.963 |
| Polar high `(4,4,3,3,2,2)` | 3.78 | 0.830 | 0.958 | 0.965 |
| `CARP-polar` | 3.35 | 0.829 | 0.957 | 0.964 |

Interpretation:

- `CARP-polar` closes essentially all of the low-to-high polar gap
- it does so at almost the low-polar bit budget
- this is the cleanest codec-side result in the repo

### 4. `q4` is the stronger base codec on this model

On the same corrected benchmark:

| Method | Bits/coord | Top-1 | Top-8 | Top-16 |
|---|---:|---:|---:|---:|
| `q4` per-channel | 4.00 | 0.863 | 0.993 | 0.998 |
| `CARP-q4-exact` | 4.24 | 0.935 | 0.968 | 0.979 |

Interpretation:

- `q4` is a stronger base codec than low-bit polar on `Qwen2.5-0.5B`
- `CARP-q4-exact` dramatically improves top-1
- but it worsens top-8/top-16 versus plain `q4`
- so the effect is “sharpen the winner”, not “preserve the whole ranking”

### 5. Real cache-path results favor `q4 -> exact` over `polar -> high_polar`

Second-step cache-path test at `512` tokens:

| Method | qasper Top-1 | multifieldqa Top-1 | 2wikimqa Top-1 |
|---|---:|---:|---:|
| `polar -> high_polar` | 0.4 | 0.8 | 1.0 |
| `q4 -> exact` | 0.8 | 1.0 | 1.0 |

And importantly:

- exact-head fallback did **not** activate at this context length
- so these runs measure promoted-token precision, not head-level exact fallback

### 6. The strongest contribution is diagnostic, not codec-theoretic

The durable contribution from this repo is:

1. A corrected offline benchmark for KV-cache codecs.
2. A failure taxonomy:
   - token-level exact fallback
   - head-level exact fallback
   - decode-step exact fallback
3. The query-trajectory divergence explanation for multi-step failures.

## Repository Layout

### Active code

- [`benchmark_real_qk_attention.py`](./benchmark_real_qk_attention.py): primary offline codec benchmark
- [`carp_cache_eval.py`](./carp_cache_eval.py): second-step cache-path benchmark
- [`carp_multistep_eval.py`](./carp_multistep_eval.py): multi-step generation stability benchmark
- [`carp_kv.py`](./carp_kv.py): selector and promotion utilities
- [`polar_quant.py`](./polar_quant.py): polar codec implementation
- [`longbench_subset_eval.py`](./longbench_subset_eval.py): retrieval-side LongBench subset experiments
- [`colab_runner.py`](./colab_runner.py): Colab-friendly command runner
- [`summarize_real_qk_benchmark.py`](./summarize_real_qk_benchmark.py): markdown summary for the real Q/K benchmark

### Legacy but still runnable

- [`profile_qwen_kv_polar.py`](./profile_qwen_kv_polar.py): legacy mixed-layer proxy profiler
- [`diagnose_qwen_kv_protocol.py`](./diagnose_qwen_kv_protocol.py): diagnosis for why the legacy proxy was wrong

Do not use these as the primary evidence for codec quality.

### Documentation

- [`docs/README.md`](./docs/README.md): documentation map
- [`docs/papers/CARP_DIAGNOSTIC_PAPER.md`](./docs/papers/CARP_DIAGNOSTIC_PAPER.md): full paper-style write-up of the project
- [`docs/reports/CARP_TECHNICAL_REPORT.md`](./docs/reports/CARP_TECHNICAL_REPORT.md): full technical report
- [`LONG_BENCH_EXPERIMENTS.md`](./LONG_BENCH_EXPERIMENTS.md): chronological experiment log used by the scripts
- [`results/MASTER_RESULTS_LOG.md`](./results/MASTER_RESULTS_LOG.md): compact results ledger
- [`notebooks/carp_gpu_colab.ipynb`](./notebooks/carp_gpu_colab.ipynb): Colab notebook wrapper around the Python runner

### Archived SRPQ work

- [`archive/srpq/README.md`](./archive/srpq/README.md)

### Third-party reference repos

- [`third_party/README.md`](./third_party/README.md)

## Experiment Timeline

### Phase 1: Retrieval on a LongBench subset

Goal:

- retrieve useful text spans for a small model under CPU-safe constraints

Outcome:

- best local retrieval result was `task_adaptive_mix_topk = 39.96`
- learned routing added little over hand-tuned routing
- this track plateaued

Primary script:

- [`longbench_subset_eval.py`](./longbench_subset_eval.py)

### Phase 2: SRPQ and codec redesign

Goal:

- build a low-rank + residual polar codec (`SRPQ`)

Outcome:

- low-rank + sparse did not work as the main codec
- the spectral machinery was more useful as a selector feature than as a codec
- these drafts are now archived

Archive:

- [`archive/srpq/`](./archive/srpq/)

### Phase 3: CARP and the benchmark correction

Goal:

- test a query-adaptive promoted subset on top of a stronger base codec

Critical discovery:

- the original key-only mixed-layer benchmark was invalid for codec ranking

Fix:

- moved to a real same-layer Q/K benchmark

Primary script:

- [`benchmark_real_qk_attention.py`](./benchmark_real_qk_attention.py)

### Phase 4: Real cache-path and multi-step evaluation

Goal:

- test whether offline codec gains survive real decoding

Findings:

- token-level and head-level behaviors differ
- multi-step failures are caused by query-trajectory divergence
- entropy-triggered exact-step fallback can stabilize generation, but can become too expensive on small models

Primary scripts:

- [`carp_cache_eval.py`](./carp_cache_eval.py)
- [`carp_multistep_eval.py`](./carp_multistep_eval.py)

## Reproducing The Main Results

### 1. Set up Colab or a GPU box

```bash
python colab_runner.py setup
```

### 2. Run the canonical offline benchmark

```bash
python colab_runner.py realbench \
  --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
  --tasks qasper multifieldqa_en 2wikimqa \
  --samples-per-task 1 \
  --max-context-tokens 1024 \
  --min-query-pos 32
```

### 3. Summarize the benchmark into a paper-ready table

```bash
python colab_runner.py summarize-realbench \
  --input results/real_qk_attention_benchmark.json \
  --output results/real_qk_attention_summary.md
```

### 4. Run the second-step cache-path comparison

Polar-backed CARP:

```bash
python colab_runner.py cache \
  --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
  --tasks qasper multifieldqa_en 2wikimqa \
  --samples-per-task 5 \
  --full-max-context-tokens 512 \
  --base-codec polar \
  --upgrade-codec high_polar \
  --selector-mode heuristic \
  --exact-head-thresholds 0.8 0.7
```

`q4`-backed CARP:

```bash
python colab_runner.py cache \
  --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
  --tasks qasper multifieldqa_en 2wikimqa \
  --samples-per-task 5 \
  --full-max-context-tokens 512 \
  --base-codec q4 \
  --upgrade-codec exact \
  --selector-mode heuristic \
  --exact-head-thresholds 0.8 0.7
```

### 5. Run the multistep benchmark

```bash
python colab_runner.py multistep \
  --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
  --tasks qasper multifieldqa_en 2wikimqa \
  --samples-per-task 1 \
  --decode-steps 8 \
  --base-codec q4 \
  --upgrade-codec exact \
  --selector-mode heuristic \
  --exact-head-risk-threshold 0.7 \
  --entropy-fallback-threshold 0.30
```

## What To Cite From This Repo

If you reuse ideas from this repo, the most defensible claims are:

- real same-layer Q/K benchmarking is necessary for KV codec evaluation
- mixed-layer key-only proxy evaluation can badly mis-rank codecs
- multi-step generation failures are best understood as query-trajectory divergence
- simple mixed-precision baselines can outperform more elaborate designs in real cache-path tests

## What Not To Overclaim

Do not claim from this repo alone that:

- a new codec beats PolarQuant on bits-quality tradeoff
- adaptive promotion is universally necessary
- polar is the best base codec across models

Those are still open questions.
