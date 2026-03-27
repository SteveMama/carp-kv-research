# Experiment Log

This file tracks benchmark runs, improvements, and comparisons for the SLM conversation-memory framework.

## 2026-03-25 15:48:33
- Notes: typed_graph_hard_benchmark_seed7_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8125, top8=0.9375
- Best generation: `qagraph_topk` exact_match=0.7500, avg_s=0.117, avg_ctx_tokens=16.6
## 2026-03-25 15:49:18
- Notes: typed_graph_hard_benchmark_seed13_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8438, top8=0.9688
- Best generation: `qagraph_topk` exact_match=0.8333, avg_s=0.154, avg_ctx_tokens=16.7
## 2026-03-25 15:50:02
- Notes: typed_graph_hard_benchmark_seed7_k4_ablation
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8125, top8=0.9375
- Best generation: `causal_full` exact_match=0.5000, avg_s=0.465, avg_ctx_tokens=749.3
## 2026-03-25 15:53:41
- Notes: typed_graph_hard_benchmark_seed21_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8438, top8=0.9688
- Best generation: `qagraph_topk` exact_match=0.7500, avg_s=0.119, avg_ctx_tokens=15.4
## 2026-03-25 16:00:05
- Notes: typed_graph_support_context_seed7_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8125, top8=0.9375
- Best generation: `qagraph_topk` exact_match=0.9167, avg_s=0.142, avg_ctx_tokens=26.0
## 2026-03-25 16:00:45
- Notes: typed_graph_support_context_seed13_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8438, top8=0.9688
- Best generation: `qagraph_topk` exact_match=0.8333, avg_s=0.178, avg_ctx_tokens=28.4
## 2026-03-25 16:01:17
- Notes: typed_graph_support_context_seed21_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8438, top8=0.9688
- Best generation: `qagraph_topk` exact_match=0.8333, avg_s=0.119, avg_ctx_tokens=24.3
## 2026-03-25 16:05:35
- Notes: typed_graph_multihop_rerank_seed7_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8125, top8=0.9375
- Best generation: `qagraph_topk` exact_match=0.9167, avg_s=0.116, avg_ctx_tokens=26.0
## 2026-03-25 16:06:14
- Notes: typed_graph_multihop_rerank_seed13_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8438, top8=0.9688
- Best generation: `qagraph_topk` exact_match=0.8333, avg_s=0.160, avg_ctx_tokens=28.4
## 2026-03-25 16:06:50
- Notes: typed_graph_multihop_rerank_seed21_k1
- Config: train=10, test=4, turns=52, queries=8
- Best retrieval: `qaware_graph` top1=0.8438, top8=0.9688
- Best generation: `qagraph_topk` exact_match=0.8333, avg_s=0.113, avg_ctx_tokens=24.3
