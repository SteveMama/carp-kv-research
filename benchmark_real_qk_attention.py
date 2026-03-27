from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from carp_cache_eval import group_query_heads
from carp_kv import (
    apply_selector_weights,
    build_topk_labels,
    compute_spectral_innovation,
    selector_feature_matrix,
)
from main import HF_QWEN_SMALL
from polar_quant import RecursivePolarQuantizer, randomized_hadamard_matrix
from profile_qwen_kv_polar import load_contexts, lowrank_sparse_reconstruct, per_channel_q4_reconstruct


RESULTS_DIR = Path(__file__).resolve().parent / "results"


_ATTN_COLLECTOR: list[dict[str, torch.Tensor]] | None = None


def patch_qwen_attention_capture() -> object:
    original = Qwen2Attention.forward

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        global _ATTN_COLLECTOR
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if _ATTN_COLLECTOR is not None:
            _ATTN_COLLECTOR.append(
                {
                    "layer_idx": torch.tensor(self.layer_idx),
                    "query_states": query_states.detach().cpu(),
                    "key_states": key_states.detach().cpu(),
                }
            )

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "query_states": query_states.detach(),
            }
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    Qwen2Attention.forward = forward
    return original


def restore_qwen_attention(original) -> None:
    Qwen2Attention.forward = original


def collect_attention_tensors(
    model,
    tokenizer,
    items: list[dict],
    max_context_tokens: int,
) -> list[dict]:
    global _ATTN_COLLECTOR
    records: list[dict] = []
    for item in items:
        prompt = item["context"]
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_tokens)
        _ATTN_COLLECTOR = []
        with torch.no_grad():
            model(**encoded, use_cache=False)
        layer_records = []
        for rec in _ATTN_COLLECTOR or []:
            q = rec["query_states"].squeeze(0).to(torch.float32)
            k = rec["key_states"].squeeze(0).to(torch.float32)
            q_grouped = group_query_heads(q.unsqueeze(0), k.shape[0]).squeeze(0)
            layer_records.append(
                {
                    "layer_idx": int(rec["layer_idx"].item()),
                    "queries": q_grouped,  # [kv_heads, seq, dim]
                    "keys": k,             # [kv_heads, seq, dim]
                }
            )
        records.append(
            {
                "task": item["task"],
                "id": item["id"],
                "length": item["length"],
                "layers": layer_records,
            }
        )
        if layer_records:
            seq_len = int(layer_records[0]["keys"].shape[1])
            print(f"captured {item['task']} id={item['id']} len={item['length']} layers={len(layer_records)} seq={seq_len}")
    _ATTN_COLLECTOR = None
    return records


def update_metric_bucket(bucket: dict[str, float], truth_idx: int, approx_scores: torch.Tensor) -> None:
    order = torch.argsort(approx_scores, descending=True)
    bucket["count"] += 1
    bucket["top1"] += float(order[0].item() == truth_idx)
    bucket["top8"] += float((order[: min(8, order.shape[0])] == truth_idx).any().item())
    bucket["top16"] += float((order[: min(16, order.shape[0])] == truth_idx).any().item())


def finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = max(int(bucket["count"]), 1)
    return {
        "top1": bucket["top1"] / count,
        "top8": bucket["top8"] / count,
        "top16": bucket["top16"] / count,
        "mean_relative_l2": bucket["rel_l2"] / count,
        "queries_evaluated": int(bucket["count"]),
    }


def fit_flat_selector_weights(
    x: np.ndarray,
    y: np.ndarray,
    label_weights: np.ndarray,
    steps: int = 500,
    lr: float = 0.08,
    l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    label_weights = label_weights.astype(np.float32)
    positive = float(y.sum())
    negative = float(len(y) - positive)
    if positive < 1:
        raise RuntimeError("No positive selector labels")
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    pos_weight = negative / positive
    sample_weight = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32) * label_weights
    for _ in range(steps):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = (probs - y) * sample_weight
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def bits_per_coord(bits_per_level: list[int], radius_bits: int, dim: int) -> float:
    levels = len(bits_per_level)
    bits_per_vector = sum(
        (2 ** (levels - level_idx)) * bits
        for level_idx, bits in enumerate(bits_per_level, start=1)
    ) + radius_bits
    return bits_per_vector / dim


def update_carp_bucket(
    bucket: dict[str, float],
    truth_idx: int,
    base_scores: torch.Tensor,
    upgraded_scores: torch.Tensor,
    selector_scores: torch.Tensor,
    base_fraction: float,
    max_fraction: float,
    pivot_k: int,
    threshold: float,
    temperature: float,
) -> None:
    num_keys = base_scores.shape[0]
    max_budget = max(1, int(round(num_keys * max_fraction)))
    base_budget = max(1, int(round(num_keys * base_fraction)))
    ordered_low = torch.sort(base_scores, descending=True).values
    pivot = min(max(1, pivot_k), len(ordered_low) - 1)
    margin = float((ordered_low[0] - ordered_low[pivot]).item())
    logit = float((threshold - margin) / max(temperature, 1e-6))
    logit = max(min(logit, 30.0), -30.0)
    ambiguity = 1.0 / (1.0 + np.exp(-logit))
    budget = base_budget + int(round((max_budget - base_budget) * ambiguity))
    budget = max(1, min(num_keys, budget))
    chosen = torch.topk(selector_scores, k=budget).indices
    mixed_scores = base_scores.clone()
    mixed_scores[chosen] = upgraded_scores[chosen]
    update_metric_bucket(bucket, truth_idx, mixed_scores)
    bucket["used_fraction"] += budget / max(num_keys, 1)
    bucket["ambiguity"] += ambiguity


def finalize_carp_bucket(bucket: dict[str, float], approx_bits_per_coord: float) -> dict[str, float]:
    count = max(int(bucket["count"]), 1)
    return {
        "top1": bucket["top1"] / count,
        "top8": bucket["top8"] / count,
        "top16": bucket["top16"] / count,
        "mean_used_fraction": bucket["used_fraction"] / count,
        "mean_ambiguity": bucket["ambiguity"] / count,
        "approx_bits_per_coord": approx_bits_per_coord,
        "queries_evaluated": int(bucket["count"]),
    }


def build_head_records(
    item_layers: list[dict],
    bits_per_level: list[int],
    high_bits_per_level: list[int],
    radius_bits: int,
    high_radius_bits: int,
    lowrank_rank: int,
    lowrank_sparse_k: int,
    spectral_rank: int,
    seed: int,
) -> list[dict]:
    head_records: list[dict] = []
    for layer in item_layers:
        keys_all = layer["keys"]
        queries_all = layer["queries"]
        kv_heads, _, head_dim = keys_all.shape
        low_q = RecursivePolarQuantizer(dim=head_dim, bits_per_level=bits_per_level, radius_bits=radius_bits)
        high_q = RecursivePolarQuantizer(dim=head_dim, bits_per_level=high_bits_per_level, radius_bits=high_radius_bits)
        for head_idx in range(kv_heads):
            keys = keys_all[head_idx]
            queries = queries_all[head_idx]
            signs = randomized_hadamard_matrix(head_dim, seed=seed + layer["layer_idx"] * 31 + head_idx)
            low_codebooks = low_q.fit_codebooks(keys, precondition_signs=signs)
            high_codebooks = high_q.fit_codebooks(keys, precondition_signs=signs)
            low_recon = low_q.dequantize(
                low_q.quantize(keys, low_codebooks, precondition_signs=signs),
                low_codebooks,
                precondition_signs=signs,
            )
            high_recon = high_q.dequantize(
                high_q.quantize(keys, high_codebooks, precondition_signs=signs),
                high_codebooks,
                precondition_signs=signs,
            )
            q4_recon = per_channel_q4_reconstruct(keys, keys)
            lowrank_recon = lowrank_sparse_reconstruct(
                keys,
                keys,
                rank=min(lowrank_rank, max(1, head_dim - 1)),
                sparse_k=min(lowrank_sparse_k, head_dim),
            )
            innovation = compute_spectral_innovation(
                keys,
                keys,
                rank=min(spectral_rank, max(1, head_dim - 1)),
            )
            head_records.append(
                {
                    "layer_idx": layer["layer_idx"],
                    "head_idx": head_idx,
                    "keys": keys,
                    "queries": queries,
                    "low_recon": low_recon,
                    "high_recon": high_recon,
                    "q4_recon": q4_recon,
                    "lowrank_recon": lowrank_recon,
                    "innovation": innovation,
                }
            )
    return head_records


def train_carp_selector(
    head_records: list[dict],
    min_query_pos: int,
    train_queries_per_head: int,
    positive_k: int,
) -> tuple[np.ndarray, float, int]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    query_count = 0
    for record in head_records:
        seq_len = record["keys"].shape[0]
        train_end = min(seq_len, min_query_pos + train_queries_per_head)
        for pos in range(max(min_query_pos, 1), train_end):
            query = record["queries"][pos]
            true_scores = (query @ record["keys"][: pos + 1].T).unsqueeze(0)
            low_scores = (query @ record["low_recon"][: pos + 1].T).unsqueeze(0)
            lowrank_scores = (query @ record["lowrank_recon"][: pos + 1].T).unsqueeze(0)
            features = selector_feature_matrix(low_scores, lowrank_scores, record["innovation"][: pos + 1])
            labels, label_weights = build_topk_labels(true_scores, positive_k=min(positive_k, pos + 1))
            x_list.append(features.reshape(-1, features.shape[-1]))
            y_list.append(labels.reshape(-1))
            w_list.append(label_weights.reshape(-1))
            query_count += 1
    if not x_list:
        raise RuntimeError("No selector training queries were collected")
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    w = np.concatenate(w_list, axis=0)
    weights, bias = fit_flat_selector_weights(x, y, w)
    return weights, bias, query_count


def evaluate_item_layers(
    item_layers: list[dict],
    bits_per_level: list[int],
    high_bits_per_level: list[int],
    radius_bits: int,
    high_radius_bits: int,
    lowrank_rank: int,
    lowrank_sparse_k: int,
    spectral_rank: int,
    selector_positive_k: int,
    base_fraction: float,
    max_fraction: float,
    pivot_k: int,
    threshold: float,
    temperature: float,
    train_queries_per_head: int,
    seed: int,
    min_query_pos: int,
) -> dict[str, dict[str, float]]:
    polar_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "rel_l2": 0.0}
    polar_high_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "rel_l2": 0.0}
    q4_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "rel_l2": 0.0}
    lowrank_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "rel_l2": 0.0}
    carp_polar_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "used_fraction": 0.0, "ambiguity": 0.0}
    carp_q4_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "used_fraction": 0.0, "ambiguity": 0.0}

    head_records = build_head_records(
        item_layers,
        bits_per_level=bits_per_level,
        high_bits_per_level=high_bits_per_level,
        radius_bits=radius_bits,
        high_radius_bits=high_radius_bits,
        lowrank_rank=lowrank_rank,
        lowrank_sparse_k=lowrank_sparse_k,
        spectral_rank=spectral_rank,
        seed=seed,
    )
    selector_weights, selector_bias, train_query_count = train_carp_selector(
        head_records,
        min_query_pos=min_query_pos,
        train_queries_per_head=train_queries_per_head,
        positive_k=selector_positive_k,
    )

    low_bits = bits_per_coord(bits_per_level, radius_bits, head_records[0]["keys"].shape[-1])
    high_bits = bits_per_coord(high_bits_per_level, high_radius_bits, head_records[0]["keys"].shape[-1])

    for record in head_records:
        keys = record["keys"]
        queries = record["queries"]
        seq_len = keys.shape[0]
        rel_polar = (
            torch.linalg.norm(keys - record["low_recon"], dim=-1) /
            torch.clamp(torch.linalg.norm(keys, dim=-1), min=1e-6)
        )
        rel_high = (
            torch.linalg.norm(keys - record["high_recon"], dim=-1) /
            torch.clamp(torch.linalg.norm(keys, dim=-1), min=1e-6)
        )
        rel_q4 = (
            torch.linalg.norm(keys - record["q4_recon"], dim=-1) /
            torch.clamp(torch.linalg.norm(keys, dim=-1), min=1e-6)
        )
        rel_lowrank = (
            torch.linalg.norm(keys - record["lowrank_recon"], dim=-1) /
            torch.clamp(torch.linalg.norm(keys, dim=-1), min=1e-6)
        )

        eval_start = max(min_query_pos + train_queries_per_head, min_query_pos, 1)
        for pos in range(eval_start, seq_len):
            query = queries[pos]
            true_scores = query @ keys[: pos + 1].T
            truth_idx = int(torch.argmax(true_scores).item())
            low_scores = query @ record["low_recon"][: pos + 1].T
            high_scores = query @ record["high_recon"][: pos + 1].T
            q4_scores = query @ record["q4_recon"][: pos + 1].T
            lowrank_scores = query @ record["lowrank_recon"][: pos + 1].T

            update_metric_bucket(polar_bucket, truth_idx, low_scores)
            update_metric_bucket(polar_high_bucket, truth_idx, high_scores)
            update_metric_bucket(q4_bucket, truth_idx, q4_scores)
            update_metric_bucket(lowrank_bucket, truth_idx, lowrank_scores)

            polar_bucket["rel_l2"] += float(rel_polar[: pos + 1].mean().item())
            polar_high_bucket["rel_l2"] += float(rel_high[: pos + 1].mean().item())
            q4_bucket["rel_l2"] += float(rel_q4[: pos + 1].mean().item())
            lowrank_bucket["rel_l2"] += float(rel_lowrank[: pos + 1].mean().item())

            features = selector_feature_matrix(
                low_scores.unsqueeze(0),
                lowrank_scores.unsqueeze(0),
                record["innovation"][: pos + 1],
            )
            selector_scores = apply_selector_weights(features, selector_weights, selector_bias)[0]

            update_carp_bucket(
                carp_polar_bucket,
                truth_idx,
                low_scores,
                high_scores,
                selector_scores,
                base_fraction=base_fraction,
                max_fraction=max_fraction,
                pivot_k=pivot_k,
                threshold=threshold,
                temperature=temperature,
            )
            update_carp_bucket(
                carp_q4_bucket,
                truth_idx,
                q4_scores,
                true_scores,
                selector_scores,
                base_fraction=base_fraction,
                max_fraction=max_fraction,
                pivot_k=pivot_k,
                threshold=threshold,
                temperature=temperature,
            )

    return {
        "polar": finalize_bucket(polar_bucket),
        "polar_high": finalize_bucket(polar_high_bucket),
        "q4": finalize_bucket(q4_bucket),
        "lowrank_sparse": finalize_bucket(lowrank_bucket),
        "carp_polar": finalize_carp_bucket(
            carp_polar_bucket,
            approx_bits_per_coord=((1.0 - (carp_polar_bucket["used_fraction"] / max(carp_polar_bucket["count"], 1.0))) * low_bits)
            + ((carp_polar_bucket["used_fraction"] / max(carp_polar_bucket["count"], 1.0)) * high_bits),
        ),
        "carp_q4_exact": finalize_carp_bucket(
            carp_q4_bucket,
            approx_bits_per_coord=((1.0 - (carp_q4_bucket["used_fraction"] / max(carp_q4_bucket["count"], 1.0))) * 4.0)
            + ((carp_q4_bucket["used_fraction"] / max(carp_q4_bucket["count"], 1.0)) * 16.0),
        ),
        "selector_training": {
            "feature_dim": int(selector_weights.shape[0]),
            "train_queries": int(train_query_count),
            "weights": selector_weights.tolist(),
            "bias": float(selector_bias),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KV key quantization against real same-layer attention queries.")
    parser.add_argument("--model-name", type=str, default=HF_QWEN_SMALL)
    parser.add_argument("--tasks", nargs="+", default=["qasper", "multifieldqa_en", "2wikimqa"])
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    parser.add_argument("--bits-per-level", type=str, default="4,3,2,2,2,2")
    parser.add_argument("--radius-bits", type=int, default=8)
    parser.add_argument("--high-bits-per-level", type=str, default="4,4,3,3,2,2")
    parser.add_argument("--high-radius-bits", type=int, default=8)
    parser.add_argument("--lowrank-rank", type=int, default=8)
    parser.add_argument("--lowrank-sparse-k", type=int, default=4)
    parser.add_argument("--spectral-rank", type=int, default=16)
    parser.add_argument("--selector-positive-k", type=int, default=8)
    parser.add_argument("--carp-base-fraction", type=float, default=0.02)
    parser.add_argument("--carp-max-fraction", type=float, default=0.18)
    parser.add_argument("--carp-pivot-k", type=int, default=8)
    parser.add_argument("--carp-threshold", type=float, default=0.2)
    parser.add_argument("--carp-temperature", type=float, default=0.08)
    parser.add_argument("--carp-train-queries-per-head", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-query-pos", type=int, default=32)
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "real_qk_attention_benchmark.json"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    model.eval()

    items = load_contexts(args.tasks, args.samples_per_task, args.max_words)
    original_forward = patch_qwen_attention_capture()
    try:
        records = collect_attention_tensors(
            model,
            tokenizer,
            items,
            max_context_tokens=args.max_context_tokens,
        )
    finally:
        restore_qwen_attention(original_forward)

    bits_per_level = [int(part) for part in args.bits_per_level.split(",") if part.strip()]
    high_bits_per_level = [int(part) for part in args.high_bits_per_level.split(",") if part.strip()]

    per_item_results = []
    for record in records:
        metrics = evaluate_item_layers(
            record["layers"],
            bits_per_level=bits_per_level,
            high_bits_per_level=high_bits_per_level,
            radius_bits=args.radius_bits,
            high_radius_bits=args.high_radius_bits,
            lowrank_rank=args.lowrank_rank,
            lowrank_sparse_k=args.lowrank_sparse_k,
            spectral_rank=args.spectral_rank,
            selector_positive_k=args.selector_positive_k,
            base_fraction=args.carp_base_fraction,
            max_fraction=args.carp_max_fraction,
            pivot_k=args.carp_pivot_k,
            threshold=args.carp_threshold,
            temperature=args.carp_temperature,
            train_queries_per_head=args.carp_train_queries_per_head,
            seed=args.seed,
            min_query_pos=args.min_query_pos,
        )
        per_item_results.append(
            {
                "task": record["task"],
                "id": record["id"],
                "length": record["length"],
                **metrics,
            }
        )

    def macro(codec: str) -> dict[str, float]:
        rows = [row[codec] for row in per_item_results]
        return {
            "top1": float(sum(row["top1"] for row in rows) / len(rows)),
            "top8": float(sum(row["top8"] for row in rows) / len(rows)),
            "top16": float(sum(row["top16"] for row in rows) / len(rows)),
            **(
                {"mean_relative_l2": float(sum(row["mean_relative_l2"] for row in rows) / len(rows))}
                if "mean_relative_l2" in rows[0]
                else {}
            ),
            **(
                {"mean_used_fraction": float(sum(row["mean_used_fraction"] for row in rows) / len(rows))}
                if "mean_used_fraction" in rows[0]
                else {}
            ),
            **(
                {"mean_ambiguity": float(sum(row["mean_ambiguity"] for row in rows) / len(rows))}
                if "mean_ambiguity" in rows[0]
                else {}
            ),
            **(
                {"approx_bits_per_coord": float(sum(row["approx_bits_per_coord"] for row in rows) / len(rows))}
                if "approx_bits_per_coord" in rows[0]
                else {}
            ),
            "queries_evaluated": int(sum(row["queries_evaluated"] for row in rows)),
        }

    result = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "min_query_pos": args.min_query_pos,
        "bits_per_level": bits_per_level,
        "radius_bits": args.radius_bits,
        "high_bits_per_level": high_bits_per_level,
        "high_radius_bits": args.high_radius_bits,
        "carp": {
            "base_fraction": args.carp_base_fraction,
            "max_fraction": args.carp_max_fraction,
            "pivot_k": args.carp_pivot_k,
            "threshold": args.carp_threshold,
            "temperature": args.carp_temperature,
            "selector_positive_k": args.selector_positive_k,
            "train_queries_per_head": args.carp_train_queries_per_head,
            "spectral_rank": args.spectral_rank,
        },
        "per_item": per_item_results,
        "macro": {
            "polar": macro("polar"),
            "polar_high": macro("polar_high"),
            "q4": macro("q4"),
            "lowrank_sparse": macro("lowrank_sparse"),
            "carp_polar": macro("carp_polar"),
            "carp_q4_exact": macro("carp_q4_exact"),
        },
        "notes": {
            "protocol": "real post-RoPE queries and keys from the same layer/head; causal key prefix per query position",
            "purpose": "replace the earlier keys-as-queries mixed-layer proxy",
            "carp_training": "selector is trained online on the first train_queries_per_head query positions of each head and evaluated on later positions",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
