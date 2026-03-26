# CARP Memory

Date: 2026-03-25

`CARP` stands for `Calibrated Adaptive Residual Polar`.

This is the final pragmatic framework after the SRPQ and low-rank-only dead ends:

1. `CARP-KV`
   - model-native KV compression
   - low-bit polar codec for all keys
   - small promoted subset upgraded using a learned selector with a spectral prior
2. `CARP-Select`
   - text-span retrieval for small-model prompting
   - learned/task-adaptive reranking over dense, BM25, graph, and qaware features

The reason for this split is empirical:

- polar quantization is the strongest KV backbone we validated on real Qwen KV vectors
- the old low-rank+sparse math is useful as a prior or feature source, but not as the core codec
- LongBench subset performance for the SLM path comes from learned retrieval selection, not from direct KV compression alone

## 1. CARP-KV

Let:

```math
\hat k_i^{\text{low}} = Q_{\text{low}}(k_i), \qquad
\hat k_i^{\text{high}} = Q_{\text{high}}(k_i)
```

where:

- `Q_low` is the low-bit polar codec
- `Q_high` is the stronger polar codec

In the current implementation:

- `Q_low = (4,3,2,2,2,2)` bits/level
- `Q_high = (4,4,3,3,2,2)` bits/level

### Spectral prior

For a calibration basis `U_r`, define token innovation:

```math
\iota_i = \frac{\|(I - U_r U_r^\top)(k_i - \mu)\|_2^2}{\|k_i - \mu\|_2^2}
```

This is not used as the codec itself. It is used as a static risk feature.

### Query-conditioned promotion score

For a query `q`, compute:

```math
s_i^{\text{low}} = q^\top \hat k_i^{\text{low}}, \qquad
s_i^{\text{lr}} = q^\top \tilde k_i^{\text{lr}}
```

where `\tilde k_i^{lr}` is the low-rank+sparse proxy.

Build a feature vector:

```math
\phi_i(q) =
\big[
z(s_i^{\text{low}}),
z(s_i^{\text{lr}}),
\Delta_i,
|\Delta_i|,
\text{rank features},
\iota_i,
z(s_i^{\text{low}})\iota_i,
\Delta_i \iota_i
\big]
```

with:

```math
\Delta_i = z(s_i^{\text{lr}}) - z(s_i^{\text{low}})
```

Then fit a lightweight selector:

```math
u_i(q) = w^\top \phi_i(q) + b
```

The positive labels are `top-K` true attention keys on a held-out calibration slice, not just the single best key.

### Mixed score

For each query, promote the top `k(q)` keys by `u_i(q)`:

```math
\hat s_i(q) =
\begin{cases}
q^\top \hat k_i^{\text{high}}, & i \in \operatorname{TopK}(u(q), k(q)) \\
q^\top \hat k_i^{\text{low}}, & \text{otherwise}
\end{cases}
```

Current margin controller:

```math
k(q) = k_0 + \alpha \cdot \text{ambiguity}(q)
```

with ambiguity computed from low-polar score margins. On the best current Qwen slice, the selector is already strong enough that the adaptive part mostly collapses to the base budget.

### Current best result

On the held-out Qwen KV/query slice:

- `Q_low`:
  - `3.3438` bits/coord
  - `top1=0.6563`, `top8=0.9961`, `top16=1.0000`
- `Q_high`:
  - `3.7813` bits/coord
  - `top1=0.6719`, `top8=1.0000`, `top16=1.0000`
- `CARP-KV`:
  - mean promoted fraction `0.0200`
  - effective budget `3.3525` bits/coord
  - `top1=0.6719`, `top8=1.0000`, `top16=1.0000`

So CARP-KV matches the stronger codec at roughly `11%` of the extra bit budget.

## 2. CARP-Select

This is the text-span retrieval path for the small model.

For each chunk, we compute features from:

- dense embedding score
- qaware score
- graph score
- BM25 score
- overlap, position, passage structure

Then we use:

```math
\hat r_i = \sigma(w^\top \psi_i + b)
```

for learned reranking on non-graph tasks, and a graph-aware hybrid fallback on graph-heavy multi-doc tasks.

Current best CPU-safe LongBench subset result on `Qwen/Qwen2.5-0.5B-Instruct`:

- `task_adaptive_mix_topk = 39.96`

This remains the strongest retrieval-side method we have validated locally.

## 3. Honest Claim

What is solid:

- polar geometry is the right codec backbone on real Qwen KV vectors
- spectral low-rank structure is useful as a risk prior, not as the main codec
- a tiny promoted subset can recover the higher-quality codec behavior
- learned/task-adaptive chunk selection is the strongest SLM-side retrieval path we have on the CPU-safe LongBench subset

What is not yet proven:

- full end-to-end LongBench parity with published PolarQuant-R numbers on `Llama-3.1-8B-Instruct`
- direct Qwen generation with CARP-KV wired into the model cache
- a theorem that the spectral prior always improves promotion quality

So the current final recommendation is:

- use `CARP-KV` for model-native KV compression experiments
- use `CARP-Select` for SLM long-context text selection
- do not use pure SRPQ or pure low-rank+sparse as the main codec
