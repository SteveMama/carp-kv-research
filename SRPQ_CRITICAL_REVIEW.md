# SRPQ Critical Review

Date: 2026-03-25

This note treats `SRPQ` as the new algorithm direction, but records the parts of the
current spec that are not yet mathematically or systems-wise sound enough to implement
as written.

## 1. What Is Strong in SRPQ

The direction is substantially better aligned with the evidence we already collected:

- use a spectral decomposition
- keep the polar codec as the main quantization backbone
- move the novelty to token-adaptive handling of the residual
- unify compression and retrieval from one representation

That is a much better starting point than trying to force low-rank+sparse to be the
main codec.

## 2. Major Issues That Need Fixing

## 2.1 Retrieval theorem is not valid as written

The current theorem:

```math
P(i^* \in I_{cand}) \ge 1 - m \exp(-\Delta^2 / (2 \sigma_{res}^2))
```

is too optimistic and not justified by the stated assumptions.

Problems:

- the residual attention terms are not independent across tokens
- the residual terms are not guaranteed Gaussian just because the residual vectors are
  randomly rotated
- the miss event for top-`m` retrieval is not a simple scalar tail event
- the union-bound sketch is not enough to derive the stated form

What we can claim safely instead:

- a heuristic variance proxy for residual score uncertainty
- or an empirical recall curve as a function of `m`

For implementation, use empirical calibration before claiming a theorem.

## 2.2 The tighter angle-concentration theorem is not proven

The claim:

```math
Var(\psi_i^{(l)}) = O(1/\sqrt{d-r}) \cdot (\sigma_{r+1}^2 / \sigma_1^2)
```

is not established by the proof sketch.

Why:

- angle variance is dimensionless, so the multiplicative scaling argument needs a more
  careful derivation
- random rotation plus truncation does not automatically yield an isotropic Gaussian
  with that factor
- the residual covariance may be flatter than the original covariance, but the exact
  scaling depends on the full tail spectrum, not just `\sigma_{r+1}^2 / \sigma_1^2`

The usable version is weaker:

- residuals are often empirically more isotropic than full keys after subspace removal
- therefore polar quantization may work better on residuals than on full vectors

That is an empirical claim we can test directly.

## 2.3 Compression ratio is overstated

The current `12.9x` claim is not yet apples-to-apples.

Missing costs:

- basis storage for `\Psi_r`
- complement basis or equivalent residual basis representation
- random rotation storage or structured transform metadata
- possible padding from `d-r` to a power of two
- coefficient scale/quantization metadata
- value-cache storage

Also:

- the table uses `d=128`, `r=16`, hence `d-r=112`
- `112` is not a power of two
- if the residual path really requires power-of-two polar recursion, padding changes
  both storage and arithmetic

So the current bits/token table is only a proxy, not a real compression ratio.

## 2.4 RoPE breaks the simple O(nr) retrieval story

This is the biggest systems issue in the current spec.

You want to decompose pre-RoPE keys because they are more low-rank. But with cached
keys:

```math
k_{cached,i} = RoPE(k_i, pos_i)
```

and attention score:

```math
q_t^\top k_{cached,i}
```

If retrieval is done in the pre-RoPE basis, the query projection becomes position
dependent:

```math
q_{sub}(pos_i) = \Psi_r^\top RoPE^{-1}(q_t, pos_i)
```

That means:

- `q_sub` is not shared across tokens
- stage-1 retrieval is no longer a simple matrix-vector multiply with one query vector
- naive computation becomes `O(n r)` with a different projected query per token and
  nontrivial constant factors

More importantly, the clean “one shared subspace query” picture breaks.

This needs one of:

1. work on post-RoPE keys directly
2. use position-bucketed bases
3. approximate RoPE interaction in a structured way

Without that change, the retrieval math is not operationally clean.

## 2.5 Background-token residual dropping is risky for margin-sensitive retrieval

The current background-token argument says:

```math
\iota_i \le \tau_l \Rightarrow \text{discard residual}
```

and then bounds average error by residual energy.

That is not enough for attention or retrieval.

Why:

- top-1 ranking depends on score margins, not average reconstruction error
- a token with small residual norm can still flip ordering if the query aligns with
  precisely that missing residual direction

We already saw this failure mode in our earlier experiments.

So dropping residuals entirely should be treated as a risky mode that requires direct
margin-based evaluation, not just norm arguments.

## 2.6 The speedup estimate is wrong

The draft says:

```math
O(nr + wd) \text{ vs } O(nd)
```

and then approximates the speedup by:

```math
n / w
```

That is not correct.

The actual factor is:

```math
\frac{nd}{nr + wd} = \frac{d}{r + wd/n}
```

For `d=128`, `r=16`, `n=65536`, `w=512`:

```math
\frac{128}{16 + 1} \approx 7.5
```

not `128x`.

That is still useful, but the current claim is materially overstated.

## 2.7 SLM support is underspecified

The current text says:

- retrieve token-level KV entries from a large model
- use them for SLM inference

But a small model cannot directly consume another model's KV cache unless:

- the models share architecture, tokenizer, dimensions, and rotary conventions
- and even then the cache is not usually interchangeable

So for SLM use, retrieval must ultimately map back to:

- source token spans
- source text chunks
- or a learned cross-model memory interface

This needs to be explicit.

## 3. What SRPQ Should Become

A corrected implementation target should be:

1. **Compression target**
   - compress one model's KV cache faithfully
2. **Retrieval target**
   - retrieve source tokens or source spans
3. **SLM target**
   - feed retrieved text, not foreign KV tensors, into the SLM

That separation avoids architectural confusion.

## 4. Minimal Revision Before Implementation

Before coding the full method, the spec should be revised to:

1. replace the strong retrieval theorem with an empirical uncertainty model
2. weaken the residual-angle theorem to an empirical hypothesis
3. correct the compression accounting
4. pick a concrete RoPE strategy:
   - preferred first option: operate on post-RoPE keys for the retrieval path
5. define the SLM interface as text/span retrieval, not raw KV transfer

## 5. Recommendation

Treat `SRPQ` as the right new family, but not yet as a finalized theorem package.

The best implementation order is:

1. **SRPQ-Compression**
   - spectral decomposition + residual polar codec
2. **SRPQ-Retrieval**
   - evaluate subspace prefiltering empirically
3. **SRPQ-SLM**
   - map selected tokens back to text spans for the smaller model

That order keeps the theory honest and makes the system buildable.
