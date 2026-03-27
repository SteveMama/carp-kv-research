# New Algorithm Spec

Date: 2026-03-25

## Working Name

`SHELL-KV`

Sparse High-Error Leverage Layer for KV compression

## Why Start From Scratch

The previous line of work taught us three things:

1. low-rank+sparse memory is not strong enough as the main KV codec
2. polar-style quantization is the stronger backbone on real Qwen KV vectors
3. selector quality is no longer the main bottleneck once mixed precision is introduced

So the new algorithm should not be:

- another retrieval heuristic
- another low-rank codec
- another selector-only improvement

The new algorithm should attack the real remaining problem:

**the gap between the cheap path and the precise path is too small**

## Core Idea

Represent each key vector as:

```math
k_i = \tilde{k}_i^{(base)} + P_{S_i}\,\delta_i + e_i
```

where:

- `\tilde{k}_i^{(base)}` is an aggressive low-bit polar code
- `P_{S_i}\,\delta_i` is a tiny exact residual shell
- `S_i` is a small set of residual coordinates or residual atoms chosen for that token
- `e_i` is the remaining residual

The difference from the previous hybrid is important:

- before, we upgraded some tokens from one polar code to a slightly better polar code
- now, we keep the low polar code for all tokens and add a **query-relevant exact residual shell** for risky tokens

This should create a larger precision gap than low-bit vs slightly-higher-bit polar alone.

## Mathematical Objective

We care about attention-score distortion:

```math
s_i(q) = q^\top k_i
```

Approximation:

```math
\hat{s}_i(q) = q^\top \tilde{k}_i^{(base)} + q_{S_i}^\top \delta_i
```

Residual score error:

```math
\epsilon_i(q) = q^\top (k_i - \tilde{k}_i^{(base)} - P_{S_i}\delta_i)
```

Bound:

```math
|\epsilon_i(q)| \le \|q_{\bar{S}_i}\|_2 \,\|r_{i,\bar{S}_i}\|_2
```

where:

- `r_i = k_i - \tilde{k}_i^{(base)}`
- `\bar{S}_i` is the complement of the stored shell coordinates

That means shell selection should minimize worst-case score distortion, not plain reconstruction error.

## Residual Shell Selection

For each token, compute the base-code residual:

```math
r_i = k_i - \tilde{k}_i^{(base)}
```

Choose a shell support:

```math
S_i = \operatorname{TopM}\big(w \odot |r_i|\big)
```

where `w` is a leverage or sensitivity weight.

Possible choices for `w`:

1. uniform:
   ```math
   w_j = 1
   ```
2. query-covariance weighted:
   ```math
   w_j = \sqrt{(C_q)_{jj}}
   ```
3. head-aware learned weighting:
   ```math
   w = g_\theta(\text{layer}, \text{head}, \text{token stats})
   ```

The second option is the first one to try. It is mathematically justified and still cheap.

## Query-Time Scoring

For a query `q`, use:

```math
\hat{s}_i(q) = q^\top \tilde{k}_i^{(base)} + q_{S_i}^\top \delta_i
```

This is still cheap because:

- the base term is dense but low-bit
- the shell term is sparse and tiny

If `|S_i| = m`, shell scoring is only `O(m)` extra per token.

## Optional Exact Anchor Bank

Some tokens should be allowed to bypass shell compression entirely:

```math
\mathcal{A} = \{i : \rho_i > \tau\}
```

where `\rho_i` is a risk score, for example:

```math
\rho_i = \alpha \|r_i\|_2 + \beta \operatorname{Var}_q[q^\top r_i] + \gamma \,\text{reuse}_i
```

Then:

- tokens in `\mathcal{A}` keep exact keys
- other tokens use base polar + residual shell

This gives a three-tier cache:

1. exact anchors
2. low polar + shell
3. low polar only

## Why This Is Different From What We Already Tried

It is not:

- uniform polar quantization
- low-rank+sparse as the main codec
- mixed precision by only changing bitwidth
- selector learning as the main lever

It is:

- a **residual-correction codec**
- with exact sparse additive shells
- optimized for score distortion rather than just reconstruction

The intended effect is to make the precise path much more different from the cheap path than the previous high-polar upgrade did.

## Storage Budget

For dimension `d=64`:

- base low polar:
  ```math
  b_{base} \approx 3.34 \text{ bits/coord}
  ```
- shell size `m`
- if shell stores:
  - `m` residual values at `b_v` bits each
  - `m` indices at about `\log_2 d = 6` bits each

additional per-coordinate budget is approximately:

```math
\frac{m(b_v + 6)}{d}
```

Example:

- `m=4`
- `b_v=8`

then:

```math
\frac{4(8+6)}{64} = 0.875 \text{ bits/coord}
```

That is expensive if applied to all tokens, but attractive if only used on a small promoted subset.

## First Practical Version

Phase 1 should be simple:

1. base codec:
   - low polar `4,3,2,2,2,2`
2. promoted token set:
   - top `p%` by risk
3. promoted representation:
   - store exact top-`m` residual coordinates
4. score:
   - base polar score plus sparse shell correction

This is enough to test whether residual shells create a larger quality jump than higher-bit polar upgrades.

## Evaluation Plan

### Stage A: Qwen KV math

Use the same evaluation harness already built in:

- [profile_qwen_kv_polar.py](/Users/pranav/Documents/New%20project/profile_qwen_kv_polar.py)

Metrics:

- relative score error
- top-1 agreement
- top-8 containment
- top-16 containment

Compare:

1. low polar
2. high polar
3. mixed low/high polar
4. `SHELL-KV`

### Stage B: LongBench subset

If Stage A shows a real jump, integrate into the CPU-safe subset runner and compare against:

- `causal_full`
- `StreamingLLMPress`
- `ExpectedAttentionPress`
- `SnapKV`
- `dense_topk`
- `task_adaptive_mix_topk`

## Success Criteria

The new algorithm is worth pursuing only if it achieves at least one of:

1. clearly better top-1 than mixed low/high polar at similar bits
2. same top-1 with lower bits
3. materially lower score distortion on held-out Qwen KV queries

If it fails that test, it should be dropped quickly.

## Immediate Next Implementation

1. add shell residual extraction on top of low polar base code
2. add sparse shell score correction in the Qwen KV profiler
3. compare against:
   - low polar
   - high polar
   - mixed low/high polar

That is the cleanest next experiment.
