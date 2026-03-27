from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from carp_cache_eval import group_query_heads
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


def evaluate_item_layers(
    item_layers: list[dict],
    bits_per_level: list[int],
    radius_bits: int,
    lowrank_rank: int,
    lowrank_sparse_k: int,
    seed: int,
    min_query_pos: int,
) -> dict[str, dict[str, float]]:
    polar_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "rel_l2": 0.0}
    q4_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "rel_l2": 0.0}
    lowrank_bucket = {"count": 0.0, "top1": 0.0, "top8": 0.0, "top16": 0.0, "rel_l2": 0.0}

    for layer in item_layers:
        keys_all = layer["keys"]
        queries_all = layer["queries"]
        kv_heads, seq_len, head_dim = keys_all.shape
        quantizer = RecursivePolarQuantizer(dim=head_dim, bits_per_level=bits_per_level, radius_bits=radius_bits)
        for head_idx in range(kv_heads):
            keys = keys_all[head_idx]
            queries = queries_all[head_idx]
            signs = randomized_hadamard_matrix(head_dim, seed=seed + layer["layer_idx"] * 31 + head_idx)

            codebooks = quantizer.fit_codebooks(keys, precondition_signs=signs)
            polar_recon = quantizer.dequantize(
                quantizer.quantize(keys, codebooks, precondition_signs=signs),
                codebooks,
                precondition_signs=signs,
            )
            q4_recon = per_channel_q4_reconstruct(keys, keys)
            lowrank_recon = lowrank_sparse_reconstruct(
                keys,
                keys,
                rank=min(lowrank_rank, max(1, head_dim - 1)),
                sparse_k=min(lowrank_sparse_k, head_dim),
            )

            rel_polar = (
                torch.linalg.norm(keys - polar_recon, dim=-1) /
                torch.clamp(torch.linalg.norm(keys, dim=-1), min=1e-6)
            )
            rel_q4 = (
                torch.linalg.norm(keys - q4_recon, dim=-1) /
                torch.clamp(torch.linalg.norm(keys, dim=-1), min=1e-6)
            )
            rel_lowrank = (
                torch.linalg.norm(keys - lowrank_recon, dim=-1) /
                torch.clamp(torch.linalg.norm(keys, dim=-1), min=1e-6)
            )

            start_pos = max(min_query_pos, 1)
            for pos in range(start_pos, seq_len):
                query = queries[pos]
                true_scores = query @ keys[: pos + 1].T
                truth_idx = int(torch.argmax(true_scores).item())
                polar_scores = query @ polar_recon[: pos + 1].T
                q4_scores = query @ q4_recon[: pos + 1].T
                lowrank_scores = query @ lowrank_recon[: pos + 1].T

                update_metric_bucket(polar_bucket, truth_idx, polar_scores)
                update_metric_bucket(q4_bucket, truth_idx, q4_scores)
                update_metric_bucket(lowrank_bucket, truth_idx, lowrank_scores)

                polar_bucket["rel_l2"] += float(rel_polar[: pos + 1].mean().item())
                q4_bucket["rel_l2"] += float(rel_q4[: pos + 1].mean().item())
                lowrank_bucket["rel_l2"] += float(rel_lowrank[: pos + 1].mean().item())

    return {
        "polar": finalize_bucket(polar_bucket),
        "q4": finalize_bucket(q4_bucket),
        "lowrank_sparse": finalize_bucket(lowrank_bucket),
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
    parser.add_argument("--lowrank-rank", type=int, default=8)
    parser.add_argument("--lowrank-sparse-k", type=int, default=4)
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

    per_item_results = []
    for record in records:
        metrics = evaluate_item_layers(
            record["layers"],
            bits_per_level=bits_per_level,
            radius_bits=args.radius_bits,
            lowrank_rank=args.lowrank_rank,
            lowrank_sparse_k=args.lowrank_sparse_k,
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
            "mean_relative_l2": float(sum(row["mean_relative_l2"] for row in rows) / len(rows)),
            "queries_evaluated": int(sum(row["queries_evaluated"] for row in rows)),
        }

    result = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "min_query_pos": args.min_query_pos,
        "bits_per_level": bits_per_level,
        "radius_bits": args.radius_bits,
        "per_item": per_item_results,
        "macro": {
            "polar": macro("polar"),
            "q4": macro("q4"),
            "lowrank_sparse": macro("lowrank_sparse"),
        },
        "notes": {
            "protocol": "real post-RoPE queries and keys from the same layer/head; causal key prefix per query position",
            "purpose": "replace the earlier keys-as-queries mixed-layer proxy",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
