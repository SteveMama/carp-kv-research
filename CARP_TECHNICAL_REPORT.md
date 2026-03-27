# CARP-KV Technical Report

## 1. Executive Summary

This report documents the full development path of `CARP-KV`, a query-adaptive KV-cache framework initially built on top of a polar quantization backbone.

The current result is:

- the early offline evidence for a polar backbone came from a legacy proxy benchmark
- the original low-rank + sparse idea did not work as the primary codec
- the useful contribution is a **precision-allocation policy**, not a new base geometry

The final local result on `Qwen/Qwen2.5-0.5B-Instruct` is:

```math
\hat k_i(q) =
\begin{cases}
Q_{\text{high}}(k_i), & i \in \mathcal P(q) \\
Q_{\text{low}}(k_i), & i \notin \mathcal P(q)
\end{cases}
```

with:

- low-polar backbone for all keys
- query-conditioned promotion
- head-level exact fallback for risky heads
- optional entropy-triggered exact-step fallback during autoregressive generation

The strongest early offline codec result was:

- low polar: `3.3438` bits/coord, `top1=0.6563`
- high polar: `3.7813` bits/coord, `top1=0.6719`
- `CARP-KV`: `3.3525` bits/coord, `top1=0.6719`

Those numbers were later traced to a legacy benchmark that used held-out keys as pseudo-queries and mixed keys across layers. They should not be treated as the final codec-quality evidence.

The strongest multi-step local stability result was:

- exact-head threshold `0.7`
- entropy fallback threshold `0.30`
- exact generation reproduced on all `3/3` eight-step prompts

But the honest cost is high on a `0.5B` model:

```math
0.4583 \cdot 16.0 + 0.5417 \cdot 5.97 \approx 10.57 \text{ bits/coord}
```

So the local system is **not yet a competitive compression point** against published fixed-rate methods like PolarQuant. What is validated here is:

- the failure taxonomy
- the correct intervention point
- the model-scale hypothesis for GPU validation
- the need to use real same-layer Q/K attention benchmarks instead of mixed-layer key-only proxies

## 1.1 Benchmark Status

The repo now distinguishes between two different offline evaluations:

- `profile_qwen_kv_polar.py`
  - legacy proxy benchmark
  - uses held-out key vectors as pseudo-queries
  - mixes keys across layers
- `benchmark_real_qk_attention.py`
  - primary codec benchmark
  - uses real post-RoPE queries and keys from the same layer and KV head
  - evaluates causal key prefixes per query position

Future codec claims should be based on the real same-layer Q/K benchmark and on end-to-end cache-path generation tests, not on the legacy proxy alone.

## 2. Problem Statement

We want to compress the KV cache while preserving model behavior during long-context autoregressive generation.

At decode step `t`, a decoder-only transformer computes:

```math
h_t = \mathrm{Transformer}(x_{<t})
```

```math
q_t = W_q h_t
```

```math
a_t = \mathrm{softmax}(q_t^\top K / \sqrt{d})
```

```math
x_t = \mathrm{decode}(a_t V)
```

The compression goal is to reduce KV storage while preserving:

```math
\mathrm{softmax}(q_t^\top \hat K / \sqrt{d}) \hat V
\approx
\mathrm{softmax}(q_t^\top K / \sqrt{d}) V
```

Published work such as PolarQuant shows that polar-coordinate quantization is a strong fixed-rate codec. The question explored here was:

Can a query-adaptive precision policy improve over a fixed-rate polar codec?

## 3. Method Evolution

### 3.1 Original Low-Rank + Sparse Idea

The first formulation used:

```math
x_i \approx U U^\top x_i + s_i
```

where:

- `U U^\top x_i` is the low-rank component
- `s_i` is a sparse innovation residual

This looked strong on synthetic retrieval tasks, especially where the benchmark matched the inductive bias:

- explicit corrections
- clean entities
- predictable aliases
- sparse surprises

But on real Qwen KV vectors this failed as a primary codec. The representation did not preserve attention-score ordering well enough.

### 3.2 Polar Backbone

The project pivoted to a polar codec, following the core insight that angle quantization with randomized preconditioning is much stronger than naive per-channel quantization.

The fixed baselines became:

- per-channel `q4`
- low polar
- high polar

### 3.3 CARP-KV

The final codec family kept the polar backbone and moved the original math into a selector role:

```math
u_i(q) = w^\top \phi_i(q) + b
```

where `\phi_i(q)` is built from:

- low-polar score
- low-rank proxy score
- score disagreement
- rank features
- spectral innovation prior

The spectral innovation feature is:

```math
\iota_i
=
\frac{\|(I-U_rU_r^\top)(k_i-\mu)\|_2^2}
{\|k_i-\mu\|_2^2}
```

Promotion uses:

```math
\mathcal P(q) = \mathrm{TopK}(u(q), k(q))
```

and the budget is adapted from a live risk score.

## 4. Final CARP-KV Formulation

### 4.1 Query-Adaptive Promotion

The promotion rule is:

```math
\hat k_i(q)=
\begin{cases}
Q_{\text{high}}(k_i), & i \in \mathcal P(q) \\
Q_{\text{low}}(k_i), & i \notin \mathcal P(q)
\end{cases}
```

with:

```math
\mathrm{risk}(q)
=
\sigma\!\left(
-\frac{\Delta_z(q)}{\tau}
+
\lambda H_z(q)
+
\beta D(q)
\right)
```

where:

- `\Delta_z(q)` = z-scored top-1/top-2 margin
- `H_z(q)` = normalized entropy of low-polar scores
- `D(q)` = disagreement between low-polar and low-rank proxy rankings

Then:

```math
k(q)=k_{\min} + (k_{\max}-k_{\min}) \, \mathrm{risk}(q)^\gamma
```

### 4.2 Head-Level Exact Fallback

The first robust cache-path fix was not token-level but head-level:

```math
\hat K_h =
\begin{cases}
K_h^{\text{exact}}, & \mathrm{risk}_h > \theta \\
K_h^{\text{CARP}}, & \text{otherwise}
\end{cases}
```

This fixed the dominant failure mode in single-step cache evaluation.

### 4.3 Query-Level Exact-Step Fallback

Multi-step decode revealed a second failure mode: autoregressive query divergence.

Once a wrong token is emitted at step `t`, the next query changes:

```math
x_t \neq x_t^* \Rightarrow h_{t+1} \neq h_{t+1}^* \Rightarrow q_{t+1} \neq q_{t+1}^*
```

This means later KL spikes are not necessarily caused by persistently bad heads. The correct intervention is at the **step level**, triggered by output uncertainty:

```math
\text{if } \bar H(\mathrm{logits}_t) > \eta,
\text{ use exact logits at step } t
```

where `\bar H` is normalized output-logit entropy and `\eta` is the fallback threshold.

## 5. Experimental Phases

### 5.1 Retrieval Track

The project first explored retrieval for small-model long-context QA. The best result on the Mac-runnable LongBench subset was:

- `task_adaptive_mix_topk = 39.96`

This beat the local baselines that were actually runnable on this hardware:

- `dense_topk = 30.58`
- `ExpectedAttentionPress = 27.97`
- `SnapKV = 23.16`

However, this track plateaued and did not become the strongest contribution.

### 5.2 Offline Qwen KV Profiling

Key results:

| Method | bits/coord | top-1 | top-8 | top-16 |
|---|---:|---:|---:|---:|
| `q4` | `4.0000` | `0.2773` | `0.7656` | `0.9336` |
| low polar | `3.3438` | `0.6563` | `0.9961` | `1.0000` |
| high polar | `3.7813` | `0.6719` | `1.0000` | `1.0000` |
| `CARP-KV` | `3.3525` | `0.6719` | `1.0000` | `1.0000` |

This was the cleanest success in the repo: high-polar behavior at almost low-polar budget.

### 5.3 Single-Step Cache-Path Evaluation

The single-step cache-path experiments on `qasper`, `multifieldqa_en`, and `2wikimqa` showed:

- token-level exact fallback did not solve the hard prompt
- uniform high polar did not solve the hard prompt
- head-level exact fallback did

The 5-sample cache-path comparison gave:

| Head threshold | median KL | worst KL | top-1 preserved | approx bits/coord |
|---|---:|---:|---:|---:|
| `0.8` | `0.0545` | `0.9420` | `14/15` | `3.60` |
| `0.7` | `0.0485` | `0.1492` | `15/15` | `5.97` |

Interpretation:

- `0.8` is cheaper but not robust
- `0.7` is the first robust local single-step operating point

### 5.4 Multi-Step Decode Stability

The 8-step generation proxy showed:

- `0.8` is unstable on hard prompts
- `0.7` is better but still not uniformly stable

This led to the query-divergence diagnosis.

### 5.5 Entropy-Triggered Exact-Step Fallback

With entropy fallback active:

- `0.8 + entropy 0.20` fixed `qasper` and `multifieldqa_en`, but not `2wikimqa`
- `0.7 + entropy 0.20` fixed all `3/3` prompts
- threshold sweep showed `0.30` preserved the same `3/3` result while reducing fallback usage

Current best multi-step local operating point:

- head threshold `0.7`
- entropy threshold `0.30`

At that setting:

- macro exact-match against exact 8-step generation: `1.0000`
- macro ROUGE-L against exact 8-step generation: `1.0000`
- mean fallback rate: `0.4583`

## 6. Failure Taxonomy

The experiments established a clear taxonomy.

### 6.1 Dead Hypothesis

Low-rank + sparse as the **primary codec** is dead on real Qwen KV vectors.

It survived only as:

- a feature source
- a selector prior

### 6.2 Wrong Local Fix

Token-level exact fallback was the wrong granularity.

The failure was distributed across tokens inside risky heads, not isolated to a few special tokens.

### 6.3 Correct Single-Step Fix

Head-level exact fallback is the right local fix for single-step cache-path robustness.

### 6.4 Correct Multi-Step Fix

The dominant multi-step failure is query-trajectory divergence. The right fix is output-level uncertainty gating, not more key-side heuristics.

## 7. Honest Cost Assessment

This is the central limitation of the local result.

At the best verified multi-step point:

- base exact-head threshold: `0.7`
- entropy fallback threshold: `0.30`
- mean fallback rate: `0.4583`

So the effective cost is:

```math
0.4583 \cdot 16.0 + 0.5417 \cdot 5.97 \approx 10.57 \text{ bits/coord}
```

This is far above:

- `PolarQuant-R`: `3.875` bits/coord
- low polar / high polar CARP offline regime: about `3.35` to `3.78`

Therefore:

- the local multi-step system is **not** a competitive compression point
- the validated contribution is a **quality-control framework**
- the cost story is likely model-scale dependent

## 8. Important Limitation of the 8-Step Proxy

The 8-step decode proxy is useful for stability analysis, but it is not a real LongBench task metric.

When the saved 8-step outputs were scored against the actual LongBench gold answers, all three variants scored `0.0`:

- exact continuation
- CARP continuation
- `0.7 + entropy 0.30` hybrid continuation

Reason:

- 8 generated tokens are not enough to produce complete task answers

So:

- the proxy is valid for diagnosing divergence
- it is not valid for final task-quality claims

## 9. What Is Proven vs Hypothesized

### Proven Locally

- low polar is much stronger than naive `q4`
- CARP-style promotion can match high-polar offline behavior at low-polar cost
- the main single-step failure is head-level
- the main multi-step failure is query-side divergence
- output-entropy gating is the right local intervention

### Still Hypothesized

- fallback rate will drop substantially on larger models
- an 8B model will be confident on more steps, reducing exact-step usage
- a larger model can make the effective bit rate competitive with PolarQuant-class methods

## 10. Why This Work Still Matters

Even though the local compression point is not competitive, the project produced:

1. A working adaptive precision framework on top of a strong polar codec
2. A clear failure taxonomy
3. A mathematically grounded explanation of when key-side vs query-side interventions are needed
4. A clean GPU validation plan

This is enough to justify further work on larger models.

## 11. GPU Validation Plan

The next meaningful experiments require GPU access.

### Immediate GPU Plan

1. Port the existing Qwen offline profiler to a larger model
2. Run 5-sample single-step cache-path evaluation on `Llama-3.1-8B-Instruct`
3. Run the 8-step generation test with:
   - `θ_head = 0.8`
   - entropy threshold calibration on a separate set
4. Evaluate task-level quality with real full-answer generation, not 8-token continuation

### Publishable Comparison Target

| Method | bits/coord | LongBench Avg |
|---|---:|---:|
| PolarQuant-R | `3.875` | `45.45` |
| CARP-KV | `~3.6` to `?` | `?` |

The central hypothesis is:

- on an 8B model, base uncertainty falls
- entropy fallback triggers much less often
- effective rate moves closer to the offline CARP regime

## 12. Current Best Local Conclusion

The strongest honest conclusion from this repo is:

- **offline codec result:** strong
- **single-step robustness result:** strong
- **multi-step quality-control result:** strong but expensive
- **competitive low-bit claim on Mac/0.5B:** not supported

The current best multi-step local operating point is:

- head threshold `0.7`
- entropy threshold `0.30`

But the current best *research* conclusion is:

`CARP-KV` is best understood as a **query-adaptive precision allocation framework on top of a polar backbone**, with a failure-aware escalation path from:

1. low-bit polar
2. promoted high-polar
3. exact risky heads
4. exact risky decode steps

That framework is now ready for GPU validation on larger models.
