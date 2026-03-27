# CARP-KV as a Diagnostic Framework for KV-Cache Quantization

## Abstract

This document reports the full experimental trajectory of a KV-cache compression project that began as a codec-design effort and ended as a diagnostic and evaluation framework for KV-cache quantization. The initial goal was to improve over fixed-rate polar quantization by promoting a small, query-conditioned subset of keys to higher precision. Early offline results appeared strong, but were later traced to an invalid benchmark that used held-out keys as pseudo-queries and mixed keys across layers. After replacing that proxy with a real same-layer query/key attention benchmark, the project reached three stable conclusions. First, real post-RoPE, same-layer, same-head, causal-prefix evaluation is necessary for meaningful offline KV-cache codec comparison. Second, quantization failures during generation are best understood at three levels: token-level, head-level, and decode-step level, with multi-step failures driven by query-trajectory divergence rather than static key distortion alone. Third, on `Qwen/Qwen2.5-0.5B-Instruct`, a simple mixed-precision baseline built from `q4` plus a tiny exact promoted subset outperforms the polar-backed variant in real cache-path fidelity, while `CARP-polar` remains a valid result under the corrected offline benchmark. The strongest contribution of the project is therefore not a new universally superior codec, but a diagnostic framework, a corrected benchmark methodology, and a set of practical mixed-precision baselines grounded in those findings.

## 1. Introduction

KV-cache compression is attractive because long-context decoding is dominated by memory footprint and memory bandwidth rather than arithmetic alone. A standard decoder-only transformer computes:

```math
h_t = \mathrm{Transformer}(x_{<t}), \qquad
q_t = W_q h_t,
```

```math
a_t = \mathrm{softmax}(q_t^\top K / \sqrt{d}), \qquad
x_t = \mathrm{decode}(a_t V).
```

The compression objective is to replace the cache with `(\hat K, \hat V)` while preserving generation quality:

```math
\mathrm{softmax}(q_t^\top \hat K / \sqrt{d}) \hat V
\approx
\mathrm{softmax}(q_t^\top K / \sqrt{d}) V.
```

The original aim of this project was straightforward:

1. find a stronger low-bit key codec,
2. use query-conditioned promotion to spend more bits only where needed,
3. validate the resulting system on real cache-path generation.

The project did produce a mixed-precision system, but the most important outcome was different: the experiments exposed failure modes and evaluation pitfalls that materially change how KV-cache quantization should be analyzed.

## 2. Research Questions

The project ultimately centered on six questions:

1. Can a low-rank plus sparse residual decomposition serve as a direct KV-cache codec?
2. Is a polar codec a better low-bit backbone than plain per-channel `q4`?
3. Can a small promoted subset recover most of the quality of a stronger codec?
4. At what granularity do KV-cache compression failures occur: token, head, or decode step?
5. Does offline score preservation predict real cache-path generation fidelity?
6. What is the right benchmark for comparing KV codecs?

## 3. Related Work

The most relevant existing directions are:

- `PolarQuant`
  - fixed-rate polar-coordinate quantization with randomized preconditioning
- `KIVI`
  - asymmetric low-bit KV quantization
- `SnapKV`, `HeadKV`, `PyramidKV`
  - pruning or retaining subsets of tokens or heads
- `KVPress`
  - query-distribution-aware KV compression and pruning
- anisotropic vector quantization (`AVQ`)
  - score distortion matters more than raw vector MSE for nearest-neighbor style use
- long-context diagnosis papers such as `Lost in the Middle` and `NoLiMa`
  - the placement and retrieval of relevant context is fragile even when all text is available

This project does not contribute a cleaner closed-form codec theorem than these works. Instead, it contributes:

1. a corrected evaluation protocol for offline codec measurement,
2. a failure taxonomy for cache-path quantization,
3. evidence that the best practical base codec can be model-dependent,
4. a simple mixed-precision promotion baseline informed by those diagnostics.

## 4. Experimental Program

The work proceeded in four phases.

### 4.1 Phase I: Retrieval on a CPU-Safe LongBench Subset

The first line of work focused on retrieval rather than direct KV compression. The target model was `Qwen/Qwen2.5-0.5B-Instruct`, and the CPU-safe evaluation subset consisted of:

- `qasper`
- `multifieldqa_en`
- `2wikimqa`

Methods evolved from dense retrieval and query-aware low-rank sketches to graph-augmented and task-adaptive retrieval pipelines. The best local retrieval result was:

```text
task_adaptive_mix_topk = 39.96
```

on the held-out subset. This was useful, but the retrieval line plateaued and did not continue improving after the routing and reranking variants stabilized.

### 4.2 Phase II: SRPQ and Low-Rank Codec Attempts

The second phase explored `SRPQ` and related low-rank + sparse residual ideas. The working hypothesis was:

```math
k_i \approx U U^\top k_i + s_i,
```

where `U U^\top k_i` captures the dominant low-rank structure and `s_i` captures sparse innovation.

This line failed as a primary codec on real Qwen KV vectors. The decomposition was mechanically valid, but it did not preserve attention-score ordering at aggressive budgets. The useful parts of this phase were later repurposed as selector features rather than as the codec itself.

### 4.3 Phase III: Polar Backbone and CARP

The project pivoted to a polar codec inspired by `PolarQuant`. This led to the `CARP` family:

```math
\hat k_i(q)=
\begin{cases}
Q_{\text{high}}(k_i), & i \in \mathcal P(q) \\
Q_{\text{low}}(k_i), & i \notin \mathcal P(q).
\end{cases}
```

The promoted set was selected by a lightweight score:

```math
u_i(q) = w^\top \phi_i(q) + b,
```

where `\phi_i(q)` included:

- low-codec score features,
- low-rank proxy score features,
- disagreement terms,
- rank features,
- a spectral innovation prior

```math
\iota_i = \frac{\|(I-U_rU_r^\top)(k_i-\mu)\|_2^2}{\|k_i-\mu\|_2^2}.
```

### 4.4 Phase IV: Real Cache-Path Diagnostics

The final phase integrated the mixed-precision path into a real cache and measured:

- second-step decode fidelity,
- multi-step generation stability,
- token-level, head-level, and step-level fallback behavior.

This phase produced the strongest diagnostic insights in the project.

## 5. Evaluation Environments

### 5.1 Local CPU / Mac Environment

The early work was constrained by a local Mac environment:

- sequential execution,
- limited memory,
- smaller sample counts,
- shorter contexts in many runs,
- practical emphasis on CPU-safe baselines and subsets.

This environment was sufficient for:

- retrieval ablations,
- early codec prototypes,
- cache-path plumbing,
- multi-step qualitative diagnostics.

It was **not** sufficient for stable broad codec claims.

### 5.2 GPU / Colab Environment

Later stages moved to GPU, mainly for:

- broader offline benchmarking,
- the corrected real same-layer Q/K benchmark,
- larger sample counts,
- repeated cache-path tests.

The GPU runs exposed where the CPU-side conclusions were robust and where they were artifacts of the earlier protocol or small held-out slices.

## 6. The Benchmark Correction

The single most important methodological result is that the original offline profiler was wrong for codec ranking.

### 6.1 The Broken Proxy

The early proxy benchmark:

- used held-out keys as pseudo-queries,
- mixed keys from all layers into one pool.

That benchmark produced wildly misleading conclusions. In one GPU run, it made low-bit polar look catastrophically bad, with top-1 near `0.10`.

### 6.2 The Corrected Benchmark

The corrected benchmark in [`benchmark_real_qk_attention.py`](../../benchmark_real_qk_attention.py) measures:

- real post-RoPE queries,
- real post-RoPE keys,
- same layer,
- same KV head,
- causal key prefixes for each query position.

This changed the picture completely.

## 7. Corrected Offline Results

On `Qwen/Qwen2.5-0.5B-Instruct`, using 138,240 real same-layer Q/K evaluations across `qasper`, `multifieldqa_en`, and `2wikimqa`, the macro results were:

| Method | Bits/coord | Top-1 | Top-8 | Top-16 | Mean Relative L2 |
|---|---:|---:|---:|---:|---:|
| Low-rank sparse | unquantized proxy | 0.802 | 0.955 | 0.979 | 0.360 |
| Polar low `(4,3,2,2,2,2)` | 3.34 | 0.809 | 0.954 | 0.963 | 0.147 |
| Polar high `(4,4,3,3,2,2)` | 3.78 | 0.830 | 0.958 | 0.965 | 0.122 |
| `CARP-polar` | 3.35 | 0.829 | 0.957 | 0.964 | n/a |
| `q4` per-channel | 4.00 | 0.863 | 0.993 | 0.998 | 0.094 |
| `CARP-q4-exact` | 4.24 | 0.935 | 0.968 | 0.979 | n/a |

These numbers support four conclusions.

### 7.1 `CARP-polar` Is Real Under the Correct Protocol

`CARP-polar` matches the high-polar variant at essentially the low-polar budget:

```text
polar low:   3.34 bits -> 0.809 top-1
polar high:  3.78 bits -> 0.830 top-1
CARP-polar:  3.35 bits -> 0.829 top-1
```

So the promoted-subset idea is valid under the corrected benchmark.

### 7.2 `q4` Is the Stronger Base Codec on This Model

On this model, `q4` outperforms low-bit polar in both top-k containment and relative L2. So the earlier “polar is clearly the right backbone” story is too strong. The best base codec is model-dependent.

### 7.3 `CARP-q4-exact` Sharpens Top-1 but Not the Whole Ranking

`CARP-q4-exact` dramatically improves top-1:

```text
q4:             0.863 top-1
CARP-q4-exact:  0.935 top-1
```

But top-8 and top-16 are worse than plain `q4`. This means the promotion improves the leading winner while over-concentrating probability mass away from some of the broader high-scoring set.

### 7.4 The Adaptive Budget Was Mostly Dormant Offline

On these offline runs, the adaptive budget barely moved. The mean ambiguity was effectively zero, and the promoted fraction stayed near the base value. So the main offline value of `CARP` was a calibrated fixed promoted subset, not the full adaptive budget logic.

## 8. Cache-Path Results

The next question was whether offline score preservation survives real decoding.

### 8.1 Polar-backed CARP

At `512` context tokens, second-step cache-path results were:

| Task | First-token match | Second-step top-1 | Baseline top-1 in CARP top-5 | Mean KL |
|---|---:|---:|---:|---:|
| `qasper` | 1.0 | 0.4 | 0.4 | 0.5001 |
| `multifieldqa_en` | 1.0 | 0.8 | 1.0 | 0.1714 |
| `2wikimqa` | 1.0 | 1.0 | 1.0 | 0.1864 |

### 8.2 q4-backed CARP

At the same context length:

| Task | First-token match | Second-step top-1 | Baseline top-1 in CARP top-5 | Mean KL |
|---|---:|---:|---:|---:|
| `qasper` | 1.0 | 0.8 | 1.0 | 0.00448 |
| `multifieldqa_en` | 1.0 | 1.0 | 1.0 | 0.0239 |
| `2wikimqa` | 1.0 | 1.0 | 1.0 | 0.00197 |

### 8.3 Interpretation

On this real second-step benchmark:

- `q4 -> exact promoted tokens` clearly beats `polar -> high_polar promoted tokens`
- the exact-head fallback never activated at `512` tokens
- so the observed gains are from promoted-token precision, not head-level exact fallback

This is one of the most important practical results in the repo.

## 9. Failure Taxonomy

The project uncovered a three-level failure taxonomy.

### 9.1 Token-Level Fallback

The first natural intervention is to make a small set of risky promoted tokens exact. This helps in some settings but did not fully explain the strongest failures.

### 9.2 Head-Level Fallback

Single-step cache-path experiments showed that some failures are better explained at the head level than the token level. Making an entire risky head exact could fix cases where token-level exact fallback did not.

### 9.3 Decode-Step Fallback

Multi-step generation exposed a deeper issue: later failures were not simply caused by persistent bad heads. Instead, they appeared as sudden, step-local KL spikes.

This led to the more correct interpretation: once generation diverges, the model follows a different query trajectory.

## 10. Query-Trajectory Divergence

The most useful conceptual result from the multi-step experiments is that errors propagate primarily through the query, not only through the compressed keys.

If compression causes a wrong token at step `t`, then:

```math
x_t \neq x_t^* \Rightarrow h_{t+1} \neq h_{t+1}^* \Rightarrow q_{t+1} \neq q_{t+1}^*.
```

Even if the later cache access were exact, it would be exact for the wrong query. This explains why multi-step KL does not behave like a steady additive drift:

```math
\mathrm{KL}_t \not\approx \mathrm{KL}_{t-1} + c.
```

Instead, local instability depends on where the autoregressive trajectory moves in hidden-state space. This motivated the step-level fallback rule based on output entropy.

## 11. Multi-Step Results and Cost

The strongest local eight-step stability result used:

- exact-head threshold `0.7`
- entropy fallback threshold `0.30`

This reproduced exact eight-step generation on the small prompt set, but at high effective cost on the `0.5B` model. Using the measured fallback rate:

```math
0.4583 \cdot 16.0 + 0.5417 \cdot 5.97 \approx 10.57 \text{ bits/coord}.
```

So the multi-step exact-fallback path is useful as a diagnostic and robustness tool, but not yet a compelling compression point on this small model.

## 12. CPU vs GPU Lessons

The project taught a clear CPU-vs-GPU lesson.

### CPU / Mac strengths

- sufficient to discover the failure taxonomy
- sufficient to build and debug the cache-path machinery
- sufficient to detect that some prompts fail in ways that aggregate metrics hide

### CPU / Mac weaknesses

- too easy to overfit to small slices
- too expensive for broad codec comparisons
- too constrained for stable conclusions about larger-model behavior

### GPU strengths

- enough scale to validate the corrected benchmark
- enough scale to reveal that the old proxy was wrong
- enough scale to compare `polar`, `q4`, and mixed-precision systems on real attention

### GPU outcome

GPU did not simply “confirm the original story.” It corrected it:

- `CARP-polar` is valid under the right benchmark
- `q4` is the stronger base codec on this model
- the strongest cache-path result at `512` tokens is `q4 -> exact`, not `polar -> high_polar`

## 13. What Failed

Several lines did not survive broader evaluation:

1. Low-rank+sparse as the primary codec.
2. The old key-only mixed-layer profiler as a trustworthy benchmark.
3. The initial claim that the project’s main value was a codec innovation.
4. The hope that adaptive budgeting would be central in the short-context GPU regime.

These failures are part of the contribution because they led directly to the corrected methodology and the more honest final framing.

## 14. What Survived

What remains strong:

1. The corrected benchmark protocol.
2. The failure taxonomy.
3. The query-divergence explanation.
4. `CARP-polar` as a valid low-cost promotion result under the correct benchmark.
5. `q4 -> exact promoted subset` as the strongest practical cache-path baseline on this model.

## 15. Limitations

This work has several clear limitations.

1. The main validated model is `Qwen/Qwen2.5-0.5B-Instruct`.
2. The strongest cache-path results are at relatively short contexts (`512` tokens) in the GPU runs.
3. The multi-step exact-fallback path is too expensive to claim a compelling compression point.
4. The strongest head-level fallback behavior was first observed under local CPU/Mac conditions and needs broader GPU validation at longer contexts.
5. There is no apples-to-apples published-table comparison against larger-model `PolarQuant-R` numbers yet.

## 16. Recommended Final Framing

The most defensible framing for this project is not:

> We invented a universally better KV-cache codec.

It is:

> We built a diagnostic and evaluation framework for KV-cache quantization, identified where and why quantization fails, corrected a major benchmark pitfall, and derived practical mixed-precision baselines from those findings.

Under that framing, the experimental story is coherent and useful.

## 17. Conclusion

This project began as a search for a stronger KV-cache codec and ended with a more valuable result: a clearer picture of how KV quantization should be evaluated and where its failures actually come from.

The main findings are:

1. Mixed-layer key-only proxy evaluation is not reliable for codec comparison.
2. Real same-layer Q/K benchmarking is necessary.
3. Quantization failures occur at token, head, and decode-step levels.
4. Multi-step failures are driven by query-trajectory divergence.
5. On `Qwen2.5-0.5B`, `q4` is the stronger base codec in real cache-path tests.
6. `CARP-polar` is still a valid corrected-benchmark result, but the strongest practical cache-path baseline is `q4` with a tiny exact promoted subset.

So the project does not end as a simple “better codec” paper. It ends as a diagnostic paper with a practical mixed-precision baseline, and that is a stronger and more honest contribution.

## Appendix A. Canonical Files

Primary code:

- [`benchmark_real_qk_attention.py`](../../benchmark_real_qk_attention.py)
- [`carp_cache_eval.py`](../../carp_cache_eval.py)
- [`carp_multistep_eval.py`](../../carp_multistep_eval.py)
- [`carp_kv.py`](../../carp_kv.py)
- [`polar_quant.py`](../../polar_quant.py)
- [`colab_runner.py`](../../colab_runner.py)

Primary logs:

- [`LONG_BENCH_EXPERIMENTS.md`](../../LONG_BENCH_EXPERIMENTS.md)
- [`results/MASTER_RESULTS_LOG.md`](../../results/MASTER_RESULTS_LOG.md)

Supporting review:

- [`docs/reviews/LITERATURE_COMPARISON.md`](../reviews/LITERATURE_COMPARISON.md)
