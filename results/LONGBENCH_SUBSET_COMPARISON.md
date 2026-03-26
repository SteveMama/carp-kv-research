# LongBench Subset Comparison

Date: 2026-03-25

## Setup

- Benchmark: LongBench v1 subset loaded from the official `data.zip`
- Tasks: `qasper`, `multifieldqa_en`, `2wikimqa`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Execution: CPU-only on MacBook, sequential task-by-task evaluation
- Safety caps:
  - `max_samples_per_task=2`
  - `max_length_words=5500`
  - `full_max_context_tokens=3072`
  - `torch_threads=4`
- Metrics:
  - Official-style QA F1 for these tasks
- Baselines:
  - `causal_full`
  - `StreamingLLMPress`
  - `ExpectedAttentionPress`
  - `SnapKV`
  - `BM25`
  - `LLMLingua-2`
  - `dense_topk`
  - `qaware_topk`
  - `qagraph_topk`
  - `adaptive_topk` (our current framework)

## Improvement Log

### v1: `longbench_subset_v1_qwen_cpu_safe`

- Macro average over the 3-task subset:
  - `causal_full`: `25.64`
  - `StreamingLLMPress`: `34.17`
  - `ExpectedAttentionPress`: `37.50`
  - `dense_topk`: `21.35`
  - `qaware_topk`: `35.71`
  - `qagraph_topk`: `34.17`
- Peak RSS: `986.3 MB`
- Failure mode:
  - `multifieldqa_en` was the bottleneck. Graph propagation over coarse single-document chunks hurt retrieval quality.

### Ablation: `multifield_ablation_top2_chunk80`

- Finding:
  - `multifieldqa_en` improved when chunking was finer and retrieval width was smaller.
  - `qaware_topk` increased to `40.66`.

### v2: `longbench_subset_v2_adaptive`

- Change:
  - added `adaptive_topk`, choosing graph retrieval on passage-style multi-doc tasks and sparse query-aware retrieval on single-doc tasks
- Result:
  - macro average `adaptive_topk`: `35.64`
- Limitation:
  - single-doc chunk size was still too coarse, so `multifieldqa_en` remained weak.

### v3: `longbench_subset_v3_adaptive_chunks`

- Change:
  - adaptive chunk sizing
  - finer chunks for single-document contexts
  - larger chunk windows retained for passage-structured multi-doc contexts
- Macro average over the 3-task subset:
  - `causal_full`: `25.64`
  - `StreamingLLMPress`: `34.17`
  - `ExpectedAttentionPress`: `37.50`
  - `dense_topk`: `36.43`
  - `qaware_topk`: `38.89`
  - `qagraph_topk`: `39.32`
  - `adaptive_topk`: `46.89`
- Peak RSS: `1317.8 MB`

### v4: `longbench_subset_v4_bm25_llmlingua`

- Change:
  - added `BM25` chunk retrieval baseline
  - added `LLMLingua-2` prompt compression baseline using `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` on CPU
- Macro average over the 3-task subset:
  - `causal_full`: `25.64`
  - `StreamingLLMPress`: `34.17`
  - `ExpectedAttentionPress`: `37.50`
  - `BM25`: `38.21`
  - `dense_topk`: `36.43`
  - `qaware_topk`: `38.89`
  - `qagraph_topk`: `39.32`
  - `adaptive_topk`: `46.89`
  - `LLMLingua-2`: `5.81`
- Peak RSS: `1886.5 MB`

Interpretation:

- `BM25` is a meaningful additional retrieval baseline and is competitive with the dense retrieval baselines on this subset.
- `LLMLingua-2` performed poorly in this setup on the subset and was much slower than retrieval-based compaction.
- `adaptive_topk` remains the strongest method among all baselines we ran locally.

### v5: `longbench_subset_v5_snapkv`

- Change:
  - added real `SnapKV` via `kvpress`
  - attempted `PyramidKV` via `kvpress`
- Macro average over the 3-task subset:
  - `causal_full`: `25.64`
  - `StreamingLLMPress`: `34.17`
  - `ExpectedAttentionPress`: `37.50`
  - `SnapKV`: `34.19`
  - `BM25`: `38.21`
  - `dense_topk`: `36.43`
  - `qaware_topk`: `38.89`
  - `qagraph_topk`: `39.32`
  - `adaptive_topk`: `46.89`
  - `LLMLingua-2`: `5.81`
- Peak RSS: `1981.7 MB`

Interpretation:

- `SnapKV` ran successfully with Qwen on this Mac subset.
- `PyramidKV` did not run successfully on this Qwen path and is currently marked unsupported in this setup.
- `adaptive_topk` still remains the strongest method among all successful local baselines.

## Current Best Result

`adaptive_topk` is the strongest method on this CPU-safe Qwen subset:

- Macro average: `46.89`
- Relative to `causal_full`: `+21.24` points
- Relative to `ExpectedAttentionPress`: `+9.39` points
- Relative to `StreamingLLMPress`: `+12.72` points
- Relative to `SnapKV`: `+12.70` points
- Relative to `BM25`: `+8.68` points
- Relative to `LLMLingua-2`: `+41.07` points

Per-task scores for `adaptive_topk`:

- `qasper`: `50.00`
- `multifieldqa_en`: `40.66`
- `2wikimqa`: `50.00`

Average context tokens for `adaptive_topk`:

- `qasper`: `187.5`
- `multifieldqa_en`: `1162.0`
- `2wikimqa`: `507.0`

Macro average context tokens:

- `causal_full`: `1742.17`
- `BM25`: `908.67`
- `ExpectedAttentionPress`: `1742.17`
- `adaptive_topk`: `618.83`
- `LLMLingua-2`: `903.17`

## Caveat Versus Published KV Compression Results

This is not directly comparable to the published LongBench-V1 compression table on `Llama-3.1-8B-Instruct`.

Differences:

- smaller base model: `Qwen2.5-0.5B-Instruct` instead of `Llama-3.1-8B-Instruct`
- subset evaluation instead of the full benchmark
- CPU-safe capped evaluation instead of full-scale benchmark execution
- our method changes memory selection before generation, not just KV compression

So the current claim is:

- on this Mac-runnable LongBench subset, the adaptive framework is stronger than the `kvpress` baselines we can run locally
- it is not yet evidence that the framework beats published methods like `PolarQuant-R (online)` on the full LongBench benchmark

## Stronger Validation

After the smaller 2-sample runs, we expanded the same subset to `5` samples per task:

- Run: `longbench_subset_v6_5samples`
- Tasks: `qasper`, `multifieldqa_en`, `2wikimqa`

Macro averages:

- `causal_full`: `24.29`
- `StreamingLLMPress`: `18.57`
- `ExpectedAttentionPress`: `22.71`
- `SnapKV`: `24.40`
- `BM25`: `27.53`
- `dense_topk`: `39.24`
- `qaware_topk`: `38.27`
- `qagraph_topk`: `29.53`
- `adaptive_topk`: `28.14`

This stronger slice changed the conclusion:

- the earlier hard-switch adaptive method was not robust
- simple dense retrieval became the strongest baseline
- so the initial positive result on the tiny slice did not fully hold up

We then tested a softer score-mixture rule:

- Run: `longbench_subset_v7_hybrid_mix`
- New method: `hybrid_mix_topk`

Macro averages:

- `dense_topk`: `39.24`
- `qaware_topk`: `38.27`
- `hybrid_mix_topk`: `30.54`
- `adaptive_topk`: `28.14`

Interpretation:

- the hybrid mixture improves materially over the brittle hard-switch adaptive rule
- but it still does not beat dense retrieval on the stronger 5-sample validation slice
- therefore the current framework is promising, but not yet validated as superior overall

## Learned Reranker

We then replaced the hand-tuned retrieval rule with a small learned reranker trained on held-out calibration samples while keeping the original math features:

- dense score
- query-aware low-rank/sparse score
- graph score
- BM25 score
- feature interactions and chunk-length signal

Run:

- `longbench_subset_v8_learned_reranker`
- `3` calibration samples per task
- `5` evaluation samples per task

Macro averages on the held-out evaluation split:

- `causal_full`: `22.45`
- `StreamingLLMPress`: `9.85`
- `ExpectedAttentionPress`: `27.97`
- `SnapKV`: `23.16`
- `BM25`: `26.42`
- `dense_topk`: `30.58`
- `qaware_topk`: `29.61`
- `qagraph_topk`: `29.05`
- `adaptive_topk`: `26.14`
- `hybrid_mix_topk`: `34.46`
- `learned_mix_topk`: `36.40`

Interpretation:

- the learned reranker is the first version of the original framework that beats dense retrieval on a stronger held-out split
- it also beats `BM25`, `SnapKV`, `ExpectedAttentionPress`, and the earlier heuristic variants
- the remaining weakness is still `2wikimqa`, where `ExpectedAttentionPress` reached `40.00` while `learned_mix_topk` reached `22.00`

So the current honest state is:

- the original math becomes competitive once the scoring layer is learned
- the framework is improving in the right direction
- but it is still not consistently dominant across all tasks

## Task-Adaptive Mix

The next refinement kept the learned reranker on single-document style tasks, but fell back to the graph-aware hybrid scorer on passage-heavy multi-document tasks.

Run:

- `longbench_subset_v10_task_adaptive`

Macro averages:

- `causal_full`: `22.45`
- `StreamingLLMPress`: `9.85`
- `ExpectedAttentionPress`: `27.97`
- `SnapKV`: `23.16`
- `BM25`: `26.42`
- `dense_topk`: `30.58`
- `qaware_topk`: `29.61`
- `qagraph_topk`: `29.05`
- `adaptive_topk`: `26.14`
- `hybrid_mix_topk`: `34.46`
- `task_adaptive_mix_topk`: `39.96`

Interpretation:

- this is the strongest result so far on the stronger held-out split
- it beats `dense_topk` by `+9.38`
- it beats `ExpectedAttentionPress` by `+11.99`
- it beats `SnapKV` by `+16.80`

So the best current version of the original framework is:

- learned reranking for non-graph tasks
- graph-aware hybrid retrieval for graph-heavy tasks
