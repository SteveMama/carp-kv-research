# Robustness Summary

## Current Sweep

Runs included:
- `typed_graph_hard_benchmark_seed7_k1`
- `typed_graph_hard_benchmark_seed13_k1`
- `typed_graph_hard_benchmark_seed21_k1`

Dataset setting:
- `train_conversations=10`
- `test_conversations=4`
- `turns_per_conversation=52`
- `queries_per_conversation=8`
- hard benchmark with `direct`, `alias`, and `multi_hop` queries

## Retrieval Mean Over 3 Seeds

| Backend | top1 mean | top1 sd | top8 mean |
|---|---:|---:|---:|
| `full` | 0.2604 | 0.0642 | 0.9479 |
| `q4` | 0.2604 | 0.0589 | 0.9479 |
| `turbo_proxy` | 0.2396 | 0.0737 | 0.9583 |
| `qaware_sparse` | 0.5104 | 0.0642 | 0.9271 |
| `qaware_graph` | 0.8333 | 0.0147 | 0.9583 |

## Generation Mean Over 3 Seeds

Model: `Qwen/Qwen2.5-0.5B-Instruct`

| Method | exact match mean | exact match sd | avg ctx tokens | avg sec |
|---|---:|---:|---:|---:|
| `causal_full` | 0.4722 | 0.0393 | 758.4 | 0.486 |
| `streaming_press` | 0.3889 | 0.1039 | 758.4 | 0.523 |
| `expected_press` | 0.2778 | 0.0393 | 758.4 | 0.636 |
| `qaware_topk` | 0.5000 | 0.0680 | 15.6 | 0.161 |
| `qagraph_topk` | 0.7778 | 0.0393 | 16.2 | 0.130 |

## `qaware_graph` Difficulty Breakdown

| Difficulty | top1 mean | top1 sd | top8 mean |
|---|---:|---:|---:|
| `direct` | 1.0000 | 0.0000 | 1.0000 |
| `alias` | 0.8242 | 0.0971 | 1.0000 |
| `multi_hop` | 0.4778 | 0.1100 | 0.7556 |

## Interpretation

- The current framework is stable on `direct` queries and strong on `alias` queries.
- `multi_hop` remains the main weakness.
- Relative to the real Mac-runnable `kvpress` baselines in this setup, `qaware_graph` is both more accurate and much more context-efficient.
- The strongest observed operating point so far is `K=1`; the `K=4` ablation underperformed because it added distractor noise.

## Latest Improvement: Typed Support Context

The latest change keeps the same retrieval scorer but changes the generation evidence assembly for `qaware_graph`:

- pick the highest-scoring answer-type turn
- add one bridge/support turn when the query pattern suggests alias or multi-hop reasoning

Three-seed comparison:

| Sweep | retrieval top1 mean | generation exact match mean | avg ctx tokens |
|---|---:|---:|---:|
| previous `qagraph_topk` | 0.8333 | 0.7778 | 16.2 |
| support-context `qagraph_topk` | 0.8333 | 0.8611 | 26.2 |

Interpretation:

- retrieval robustness did not move
- answer quality improved materially
- the cost increase is modest relative to the full-context baseline

## Multi-hop Work Since Then

Two follow-up directions were tested:

1. stronger cross-type reranking inside `qaware_graph`
2. train-split calibration of `bridge_scale` and `subject_scale`

Current outcome:

- the explicit multi-hop rerank mechanism preserves the current best performance but does not improve the three-seed mean over the support-context version
- the train-split calibration consistently selected conservative weights (`bridge_scale=0.8`, `subject_scale=1.0`) and produced effectively the same held-out results as the current best setup

Interpretation:

- the remaining bottleneck is not just scalar weighting
- the retrieval stack is already finding the right neighborhood
- the biggest remaining opportunity is better bridge-entity extraction or a small learned reranker rather than another fixed manual bonus

