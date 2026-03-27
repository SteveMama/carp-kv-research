# Literature Comparison

This note anchors the current framework against the relevant research directions.

## Relevant Literature

- [TurboQuant (ICLR 2026)](https://openreview.net/forum?id=tO3ASKZlok)
  - Focus: online vector quantization with near-optimal distortion for KV cache compression.
  - Key point: optimizes vector distortion and inner-product distortion.
- [Expected Attention / KVPress](https://huggingface.co/blog/nvidia/kvpress)
  - Focus: prune KV pairs using an estimate of future-query attention.
  - Key point: query-distribution-aware KV compression without task-specific training.
- [SnapKV](https://arxiv.org/abs/2404.14469)
  - Focus: token-level KV pruning using observed attention structure.
- [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://proceedings.mlr.press/v119/guo20h.html)
  - Focus: penalize score-direction distortion more than orthogonal reconstruction error.
  - Key point: raw MSE is not the right objective when ranking by inner product matters.
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)
  - Focus: long-context models fail to retrieve information reliably from large prompts.
- [NoLiMa](https://openreview.net/forum?id=0OshX1hiSa)
  - Focus: harder long-context retrieval when lexical overlap is removed.

## Where Our Framework Fits

The current method is not a pure KV codec. It is a retrieval-and-selection layer for SLMs:

1. query-aware low-rank sketch
2. sparse innovation channel
3. typed entity graph over turns
4. query-text seeding
5. target-type bias before final retrieval

This places it closer to a memory-routing framework than a direct quantization method.

## What Is Novel Relative to These Papers

- Relative to TurboQuant / AVQ:
  - those works optimize vector geometry and score distortion
  - our method optimizes conversational retrieval structure
- Relative to Expected Attention / SnapKV:
  - those works prune KV based on token importance
  - our method builds a typed relation graph across turns and uses it to resolve aliases and multi-hop references
- Relative to Lost in the Middle / NoLiMa:
  - those works diagnose retrieval failures
  - our benchmark and framework explicitly target those failure modes

## Current Claim Boundary

What we can defend now:
- on the implemented hard benchmark, the framework outperforms the real Mac-runnable `kvpress` baselines we tested
- the math is behaving consistently across multiple seeds
- the framework is strongest on direct and alias retrieval, weaker on multi-hop

What we cannot defend yet:
- that it beats TurboQuant itself
- that it is state-of-the-art on public long-context benchmarks
- that the novelty is fully established against all memory/retrieval literature

