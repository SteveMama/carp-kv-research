# We Tried to Build a Better KV-Cache Codec. What We Actually Found Was More Useful.

When this project started, the goal was simple:

> compress the KV cache better than the obvious baselines, keep attention quality high, and validate it on real generation.

That is not exactly where we ended up.

What we built turned out to be more valuable in a different way:

- a corrected benchmark for KV-cache codecs
- a failure taxonomy for KV quantization
- a practical mixed-precision baseline
- and a clearer explanation of why many KV-cache methods fail silently during generation

This article is the story of that project, the math that mattered, what failed, what survived, and why the final conclusion is not “we found the best codec.”

## The Original Goal

A decoder-only transformer computes attention using:

```math
h_t = \mathrm{Transformer}(x_{<t}), \qquad
q_t = W_q h_t,
```

```math
a_t = \mathrm{softmax}(q_t^\top K / \sqrt{d}), \qquad
x_t = \mathrm{decode}(a_t V).
```

KV-cache compression tries to replace `(K, V)` with `(\hat K, \hat V)` while keeping generation behavior similar:

```math
\mathrm{softmax}(q_t^\top \hat K / \sqrt{d}) \hat V
\approx
\mathrm{softmax}(q_t^\top K / \sqrt{d}) V.
```

The early idea in this repo was:

1. build a strong low-bit key codec,
2. learn which keys deserve extra precision,
3. upgrade only those keys at query time.

That mixed-precision idea became `CARP`.

## Phase 1: Before CARP, We Tried Retrieval

The repo did not begin with direct KV compression. It started with retrieval for a small model on a CPU-safe LongBench subset:

- `qasper`
- `multifieldqa_en`
- `2wikimqa`

The best result there was:

```text
task_adaptive_mix_topk = 39.96
```

This part mattered because it taught us something important: the original low-rank math was often more useful as a feature generator than as the main mechanism itself.

That lesson came back later in the KV work.

## Phase 2: The Low-Rank Codec Failed

The first real codec attempt was low-rank plus sparse residuals:

```math
k_i \approx U U^\top k_i + s_i.
```

The hope was:

- low-rank structure captures most of the key
- a small sparse residual keeps the important corrections

This sounds elegant, but on real Qwen KV vectors it did not hold up as a direct codec. The attention-score ordering degraded too much at aggressive budgets.

That was the first major pivot:

> low-rank + sparse was not the right primary codec

But it still turned out to be useful as a selector prior later.

## Phase 3: Polar Quantization Looked Promising

We then moved to a polar codec inspired by `PolarQuant`.

The rough idea is:

1. apply randomized preconditioning,
2. recursively convert vectors into a radius plus angles,
3. quantize the angles with more bits at coarse levels and fewer bits at deeper levels.

This gave a family of codecs like:

- low polar: `(4,3,2,2,2,2)`
- high polar: `(4,4,3,3,2,2)`

Then came the promotion idea:

```math
\hat k_i(q)=
\begin{cases}
Q_{\text{high}}(k_i), & i \in \mathcal P(q) \\
Q_{\text{low}}(k_i), & i \notin \mathcal P(q).
\end{cases}
```

That is the core `CARP` idea:

- everyone gets the cheap codec
- a tiny query-conditioned subset gets the better codec

The promoted set is selected by a lightweight score:

```math
u_i(q)=w^\top \phi_i(q)+b,
```

where `\phi_i(q)` includes:

- low-codec score features
- low-rank proxy score features
- disagreement features
- rank features
- a spectral innovation prior

The spectral innovation prior is:

```math
\iota_i=\frac{\|(I-U_rU_r^\top)(k_i-\mu)\|_2^2}{\|k_i-\mu\|_2^2}.
```

So the low-rank math did survive, but not as the codec. It survived as a feature.

## Then We Found a Serious Benchmarking Problem

This was the biggest methodological surprise in the project.

The early offline profiler looked at:

- held-out key vectors as pseudo-queries
- mixed keys from all layers in one giant pool

That turned out to be the wrong benchmark for codec quality.

Why?

Because real attention does **not** do that. Real attention compares:

- real post-RoPE queries
- against keys from the **same layer**
- and the **same KV head**
- over the **causal prefix**

Once we fixed the benchmark, the codec ranking changed dramatically.

## The Correct Benchmark

The correct offline benchmark in this repo is now:

- real post-RoPE `Q`
- real post-RoPE `K`
- same layer
- same head
- causal key prefix

That benchmark is implemented in:

- `benchmark_real_qk_attention.py`

And under that benchmark, the main macro results on `Qwen/Qwen2.5-0.5B-Instruct` became:

| Method | Bits/coord | Top-1 | Top-8 | Top-16 |
|---|---:|---:|---:|---:|
| Polar low | 3.34 | 0.809 | 0.954 | 0.963 |
| Polar high | 3.78 | 0.830 | 0.958 | 0.965 |
| `CARP-polar` | 3.35 | 0.829 | 0.957 | 0.964 |
| `q4` per-channel | 4.00 | 0.863 | 0.993 | 0.998 |
| `CARP-q4-exact` | 4.24 | 0.935 | 0.968 | 0.979 |

Two things matter here.

### First: `CARP-polar` is real

This is the cleanest result:

```text
polar low:   3.34 bits -> 0.809 top-1
polar high:  3.78 bits -> 0.830 top-1
CARP-polar:  3.35 bits -> 0.829 top-1
```

That means the promoted-subset idea works. A tiny promoted subset can recover almost all of the gap between low polar and high polar.

### Second: `q4` is actually the stronger base codec on this model

This was unexpected if you only looked at the earlier proxy. On this model:

- `q4` beats low-bit polar
- and `q4 + exact promoted subset` is the strongest practical result in the repo

So the honest conclusion is not:

> polar is always the right backbone

It is:

> the mixed-precision promotion idea is portable, but the best base codec is model-dependent

## The Most Important Limitation We Found

This project made one thing very clear:

> reconstruction error is not attention error

Classical codec analysis naturally gives a bound on:

```math
\|k - \hat k\|_2^2.
```

But attention does not use `L2`. It uses:

```math
q^\top k / \sqrt{d},
```

and then softmax.

If reconstruction error is on the order of `\varepsilon \|k\|^2`, then score error is only approximately controlled by:

```math
\Delta s \approx O\!\left(\varepsilon \cdot \|q\| \cdot \|k\| / \sqrt{d}\right).
```

And even that is not the full story, because softmax is nonlinear:

- near decision boundaries, small score shifts can flip the winner
- in the tails, much larger errors may barely matter

This is not just theory. We saw it in the results.

Under the corrected benchmark:

- `polar` had worse relative L2 than `q4`
  - `0.147` vs `0.094`
- but its attention top-1 was still competitive
  - `0.809` vs `0.863`

Why? Likely because the error structure matters:

- `polar` introduces structured geometric errors by shifting directions
- `q4` introduces mostly per-channel independent noise

That difference is mostly invisible in plain vector reconstruction metrics, but it matters to softmax.

This is one of the core lessons of the project.

## Cache-Path Results: Offline Is Not Enough

The next question was whether these offline results survive a real cache path.

We tested second-step decoding with the actual model cache.

At `512` context tokens:

### Polar-backed CARP

| Task | Top-1 | Top-1 in Top-5 | Mean KL |
|---|---:|---:|---:|
| `qasper` | 0.4 | 0.4 | 0.5001 |
| `multifieldqa_en` | 0.8 | 1.0 | 0.1714 |
| `2wikimqa` | 1.0 | 1.0 | 0.1864 |

### q4-backed CARP

| Task | Top-1 | Top-1 in Top-5 | Mean KL |
|---|---:|---:|---:|
| `qasper` | 0.8 | 1.0 | 0.00448 |
| `multifieldqa_en` | 1.0 | 1.0 | 0.0239 |
| `2wikimqa` | 1.0 | 1.0 | 0.00197 |

That made the practical conclusion much sharper:

> on this model and this cache-path test, `q4 -> exact promoted tokens` is the strongest practical system

## The Failure Taxonomy

The most original part of this project may be the failure taxonomy.

We found that quantization failures happen at three levels:

### 1. Token-level

The first idea is simple:

- keep a few risky promoted tokens exact

Sometimes this helps, but it does not explain the hardest failures.

### 2. Head-level

Some failures are better explained at the head level:

- a whole head becomes unstable enough that token-level exact fallback is not enough

This mattered in several single-step cache-path tests.

### 3. Decode-step level

Multi-step generation revealed the deepest failure mode:

- later KL spikes are step-local and unpredictable
- they are not well explained by “this head is always bad”

That led to the key insight:

## Query-Trajectory Divergence

Once generation emits the wrong token, the next query changes:

```math
x_t \neq x_t^* \Rightarrow h_{t+1} \neq h_{t+1}^* \Rightarrow q_{t+1} \neq q_{t+1}^*.
```

This means later exact attention is exact for the **wrong query trajectory**.

That explains why multi-step KL is not a steady linear drift:

```math
\mathrm{KL}_t \not\approx \mathrm{KL}_{t-1}+c.
```

Instead, some steps are locally stable and others are not, depending on where the model moves in hidden-state space.

This also explains why entropy-triggered exact-step fallback is only partially helpful:

- if it catches the first dangerous step, it can help
- if it misses the first branch point, later exact steps cannot fully recover

## The Multistep Result That Matters

On the GPU, we ran an 8-step multistep test for:

- `polar -> high_polar`
- heuristic selector
- exact-head threshold `0.7`
- entropy fallback threshold `0.30`

The result was mixed:

| Task | CARP Token Match | Hybrid Token Match |
|---|---:|---:|
| `qasper` | 1.00 | 1.00 |
| `multifieldqa_en` | 0.25 | 0.25 |
| `2wikimqa` | 1.00 | 1.00 |

And the hard sample was revealing:

- `multifieldqa_en` diverged at step `3`
- exact-head fallback was active
- entropy fallback fired on most steps
- and it still did not recover the exact trajectory

That is a direct empirical confirmation of the query-divergence story.

## CPU vs GPU: Why Both Mattered

This project needed both environments.

### CPU / Mac

Useful for:

- retrieval experiments
- early codec prototyping
- plumbing the cache-path machinery
- observing qualitative failures quickly

Bad for:

- broad codec comparisons
- stable multi-sample conclusions

### GPU / Colab

Necessary for:

- the corrected real Q/K benchmark
- meaningful multi-sample cache-path comparisons
- seeing that some CPU-side conclusions were artifacts of the old proxy

The GPU runs did not just make the earlier story stronger. They corrected it.

## So What Did We Actually Build?

If you describe this project as:

> a new KV-cache codec that beats prior work

that would be too strong.

If you describe it as:

> a diagnostic and evaluation framework for KV-cache quantization, plus a mixed-precision baseline that falls out of that analysis

that is accurate.

The final honest summary is:

1. The original mixed-layer key-only proxy was wrong.
2. The right offline benchmark is real same-layer Q/K attention.
3. `CARP-polar` is a valid low-cost promoted-subset result under that corrected benchmark.
4. `q4` is the stronger practical base codec on `Qwen2.5-0.5B`.
5. `q4 + exact promoted subset` is the strongest practical cache-path result in this repo.
6. Reconstruction error alone is not the right way to judge a KV codec.
7. Multi-step failures are driven by query-trajectory divergence.

That is a more useful conclusion than the original codec-first story.

## Where This Leaves the Project

The next real experiments are clear:

1. deduplicate truncated samples so repeated prefixes do not distort the summaries
2. rerun the cache-path comparisons at longer contexts
3. compare `polar -> high_polar` and `q4 -> exact` under those longer contexts
4. test a larger model, where fallback and ambiguity behavior may look very different

But even before those runs, the project already has a defensible contribution:

- a corrected benchmark
- a failure taxonomy
- a theory-backed interpretation of multi-step failure
- and a practical mixed-precision baseline

That is a strong outcome. It is just not the one we thought we were building at the start.
