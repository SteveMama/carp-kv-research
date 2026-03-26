# Qwen KV Math Comparison

Date: 2026-03-25

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Data source: LongBench subset contexts from:
  - `qasper`
  - `multifieldqa_en`
  - `2wikimqa`
- Sampling:
  - `1` item per task
  - `768` context-token cap per item
  - `1024` KV vectors per item-layer slice cap
- Head dimension: `64`
- Total collected key vectors: `73,728`
- Calibration vectors: `4,096`
- Held-out evaluation:
  - keys: `2,048`
  - queries: `256`

## Qwen Angle Concentration

After randomized Hadamard preconditioning, the recursive polar angles concentrate much more tightly around `pi/4` at deeper levels:

- level 3 std: `0.2356`
- level 4 std: `0.1554`
- level 5 std: `0.1114`

This matches the qualitative claim behind PolarQuant-style geometry: higher recursion levels are much easier to quantize aggressively.

## Initial Bug Fixes

The first implementation had two structural bugs:

1. The inverse of the randomized Hadamard preconditioner was wrong.
   - forward used `H D`
   - inverse was incorrectly also using `H D`
   - the correct inverse is `D H`
2. The first polar angle was quantized as an ordinary scalar instead of a circular variable.

After fixing those:

- exact precondition/decondition MSE dropped to numerical noise
- real-Qwen calibration relative L2 error dropped from about `1.386` to `0.152`

## Held-Out KV Approximation Comparison

All numbers below are on the same held-out Qwen KV/query sample.

### Polar-style quantization sweep

| Bits/level | Bits/coord | Rel L2 | Rel score err | Top-1 | Top-8 | Top-16 |
|---|---:|---:|---:|---:|---:|---:|
| `4,3,2,2,2,2` | `3.3438` | `0.1945` | `0.2223` | `0.4609` | `0.9063` | `0.9727` |
| `4,4,2,2,2,2` | `3.5938` | `0.1893` | `0.2157` | `0.4648` | `0.9258` | `0.9805` |
| `4,4,3,2,2,2` | `3.7188` | `0.1781` | `0.2026` | `0.4727` | `0.9570` | `0.9922` |
| `4,4,3,3,2,2` | `3.7813` | `0.1727` | `0.1958` | `0.5117` | `0.9492` | `0.9922` |

### Simpler baselines

| Method | Budget note | Rel L2 | Rel score err | Top-1 | Top-8 | Top-16 |
|---|---|---:|---:|---:|---:|---:|
| `q4_per_channel` | plain per-channel 4-bit | `0.4425` | `0.5192` | `0.1797` | `0.6016` | `0.7500` |
| `lowrank_sparse` | rank `8`, sparse `4` | `0.7587` | `0.8110` | `0.0234` | `0.0898` | `0.1484` |

## Low-Rank + Sparse Sweep

The original low-rank+sparse math improves with much larger rank and residual budgets, but it stays behind the polar codec at comparable or lower effective budgets.

| Rank | Sparse k | Rel L2 | Rel score err | Top-1 | Top-8 | Top-16 |
|---:|---:|---:|---:|---:|---:|---:|
| `8` | `4` | `0.7587` | `0.8110` | `0.0234` | `0.0898` | `0.1484` |
| `16` | `4` | `0.7105` | `0.7838` | `0.0313` | `0.1719` | `0.2539` |
| `16` | `8` | `0.5942` | `0.7063` | `0.0664` | `0.2656` | `0.3945` |
| `24` | `8` | `0.5334` | `0.6207` | `0.1797` | `0.5039` | `0.6289` |
| `32` | `8` | `0.4776` | `0.5185` | `0.2969` | `0.7109` | `0.8477` |
| `32` | `16` | `0.3085` | `0.3395` | `0.3555` | `0.8320` | `0.9375` |

Important caveat:

- these low-rank+sparse numbers do not include coefficient quantization
- a configuration like rank `32`, sparse `16` carries a much larger storage footprint than the `3.7` to `3.8` bits/coord polar settings

## Interpretation

Current conclusion:

- the original low-rank+sparse math is useful as a retrieval feature space
- it is not competitive as a direct KV codec at aggressive budgets
- the polar path is already materially stronger on real Qwen KV score preservation

What this means for the framework:

- if the goal is LongBench-style KV compression quality, the quantization backbone should be polar-style
- if the goal is a differentiated system, the novelty should move to precision allocation, routing, or a retrieval-conditioned mixed-precision layer on top of that backbone

## Mixed-Precision Hybrid Test

We then tested the hybrid idea:

- base codec: low-bit polar `4,3,2,2,2,2`
- upgraded codec: higher-bit polar `4,4,3,3,2,2`
- selection policy: query-adaptive top-fraction of keys upgraded per query
- selector: the original low-rank+sparse approximation

### Baseline and upgraded polar

| Method | Bits/coord | Top-1 | Top-8 | Top-16 |
|---|---:|---:|---:|---:|
| low polar `4,3,2,2,2,2` | `3.3438` | `0.4609` | `0.9063` | `0.9727` |
| high polar `4,4,3,3,2,2` | `3.7813` | `0.5117` | `0.9492` | `0.9922` |

### Mixed-precision with low-rank selector

Best observed operating points:

| Selector | Fraction upgraded | Bits/coord | Top-1 | Top-8 | Top-16 |
|---|---:|---:|---:|---:|---:|
| rank `8`, sparse `4` | `0.10` | `3.3875` | `0.5000` | `0.9297` | `0.9805` |
| rank `16`, sparse `8` | `0.10` | `3.3875` | `0.5234` | `0.9453` | `0.9844` |
| rank `16`, sparse `8` | `0.20` | `3.4313` | `0.5195` | `0.9531` | `0.9922` |
| rank `32`, sparse `16` | `0.05` | `3.3656` | `0.5117` | `0.9492` | `0.9922` |

Interpretation:

- mixed precision does help over uniform low-bit polar
- the gain is real even at very small extra budget
- however, the selector saturates quickly
- stronger selectors mostly improve top-8 / top-16 stability, but they do not push top-1 far beyond the uniform high-polar result

So the current hybrid is useful, but it is not yet a decisive leap over better uniform polar settings.

## Learned Selector Check

We then replaced the heuristic selector with a small learned selector trained on a separate held-out KV/query slice.

Setup:

- selector training keys: `1024`
- selector training queries: `128`
- eval keys: `2048`
- eval queries: `256`
- feature set:
  - low-polar score z-score
  - low-rank selector score z-score
  - score delta and absolute delta
  - score product
  - low-polar and low-rank rank features

On that split:

| Method | Bits/coord | Top-1 | Top-8 | Top-16 |
|---|---:|---:|---:|---:|
| low polar `4,3,2,2,2,2` | `3.3438` | `0.6563` | `0.9961` | `1.0000` |
| high polar `4,4,3,3,2,2` | `3.7813` | `0.6719` | `1.0000` | `1.0000` |
| mixed + low-rank selector | `3.3875` | `0.6719` | `1.0000` | `1.0000` |
| mixed + learned selector | `3.3875` | `0.6719` | `1.0000` | `1.0000` |
| mixed + oracle selector | `3.3875` | `0.6719` | `1.0000` | `1.0000` |

Interpretation:

- the learned selector did not beat the heuristic selector
- the oracle also did not beat them at this budget point
- so the remaining bottleneck is not selector quality
- the bottleneck is the limited gap between the low-bit and high-bit polar codecs on the already-good candidate set

This is a useful negative result:

- selector learning alone is not the next major lever
- the next lever should be codec design, more aggressive bit separation, or a different high-precision fallback mechanism

## CARP-KV Final Consolidation

We then replaced the earlier fixed-fraction selector story with a calibrated mixed-precision policy:

- low codec: polar `4,3,2,2,2,2`
- promoted codec: polar `4,4,3,3,2,2`
- selector target: true `top-8` attention keys on a held-out slice
- selector features:
  - low-polar score features
  - low-rank proxy score features
  - static spectral innovation prior
- promotion budget:
  - base promoted fraction with optional margin-adaptive widening

This consolidated algorithm is now called `CARP-KV`:

- `Calibrated Adaptive Residual Polar KV`

### Best current operating point

On the same held-out Qwen slice:

| Method | Bits/coord | Mean promoted fraction | Top-1 | Top-8 | Top-16 | Rel score err |
|---|---:|---:|---:|---:|---:|---:|
| low polar `4,3,2,2,2,2` | `3.3438` | `0.0000` | `0.6563` | `0.9961` | `1.0000` | `0.1876` |
| high polar `4,4,3,3,2,2` | `3.7813` | `1.0000` | `0.6719` | `1.0000` | `1.0000` | `0.1691` |
| mixed + learned selector (`10%`) | `3.3875` | `0.1000` | `0.6719` | `1.0000` | `1.0000` | `0.1857` |
| `CARP-KV` | `3.3525` | `0.0200` | `0.6719` | `1.0000` | `1.0000` | `0.1874` |

### Interpretation

- `CARP-KV` matches the stronger high-polar top-1 / top-8 / top-16 behavior on this held-out slice
- it does so while upgrading only about `2%` of keys
- it keeps the effective budget only `0.0088` bits/coord above the low codec and `0.4287` bits/coord below the high codec

This is now the best practical codec story we have:

- polar quantization remains the backbone
- the old low-rank+sparse math survives as a spectral prior and feature generator
- the differentiated piece is the calibrated promotion policy, not the raw codec alone
