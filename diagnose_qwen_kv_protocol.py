from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import HF_QWEN_SMALL
from polar_quant import RecursivePolarQuantizer, randomized_hadamard_matrix
from profile_qwen_kv_polar import (
    load_contexts,
    per_channel_q4_reconstruct,
    lowrank_sparse_reconstruct,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def gather_key_vectors_per_item(
    model,
    tokenizer,
    items: list[dict],
    max_context_tokens: int,
    max_samples_per_item: int,
) -> list[dict]:
    outputs: list[dict] = []
    for item in items:
        prompt = item["context"]
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_tokens)
        with torch.no_grad():
            model_out = model(**encoded, use_cache=True)
        per_layer = []
        for layer_idx, layer_kv in enumerate(model_out.past_key_values):
            key_states = layer_kv[0].detach().cpu().squeeze(0)
            sample = key_states.reshape(-1, key_states.shape[-1])
            if sample.shape[0] > max_samples_per_item:
                sample = sample[:max_samples_per_item]
            per_layer.append(sample.to(torch.float32))
        all_vectors = torch.cat(per_layer, dim=0)
        outputs.append(
            {
                "task": item["task"],
                "id": item["id"],
                "length": item["length"],
                "vectors": all_vectors,
                "layers": per_layer,
            }
        )
        print(
            f"profiled {item['task']} id={item['id']} len={item['length']} "
            f"layers={len(per_layer)} vectors={all_vectors.shape[0]}"
        )
    return outputs


def topk_metrics(true_scores: torch.Tensor, approx_scores: torch.Tensor) -> dict:
    truth = true_scores.argmax(dim=1)
    approx_order = torch.argsort(approx_scores, dim=1, descending=True)
    return {
        "top1": (approx_order[:, 0] == truth).float().mean().item(),
        "top8": (approx_order[:, :8] == truth.unsqueeze(1)).any(dim=1).float().mean().item(),
        "top16": (approx_order[:, :16] == truth.unsqueeze(1)).any(dim=1).float().mean().item(),
    }


def evaluate_codec(
    name: str,
    calibration: torch.Tensor,
    eval_keys: torch.Tensor,
    eval_queries: torch.Tensor,
    quantizer: RecursivePolarQuantizer,
    signs: torch.Tensor,
    lowrank_rank: int,
    lowrank_sparse_k: int,
) -> dict:
    if name == "polar":
        codebooks = quantizer.fit_codebooks(calibration, precondition_signs=signs)
        approx = quantizer.dequantize(
            quantizer.quantize(eval_keys, codebooks, precondition_signs=signs),
            codebooks,
            precondition_signs=signs,
        )
    elif name == "q4":
        approx = per_channel_q4_reconstruct(calibration, eval_keys)
    elif name == "lowrank_sparse":
        approx = lowrank_sparse_reconstruct(
            calibration,
            eval_keys,
            rank=lowrank_rank,
            sparse_k=lowrank_sparse_k,
        )
    else:
        raise ValueError(f"Unknown codec {name}")

    true_scores = eval_queries @ eval_keys.T
    approx_scores = eval_queries @ approx.T
    rel_l2 = (
        torch.linalg.norm(eval_keys - approx, dim=-1) /
        torch.clamp(torch.linalg.norm(eval_keys, dim=-1), min=1e-6)
    ).mean().item()
    result = topk_metrics(true_scores, approx_scores)
    result["mean_relative_l2"] = rel_l2
    return result


def slice_for_eval(vectors: torch.Tensor, n_keys: int, n_queries: int) -> tuple[torch.Tensor, torch.Tensor]:
    needed = n_keys + n_queries
    if vectors.shape[0] < needed:
        raise RuntimeError(f"Need at least {needed} vectors, found {vectors.shape[0]}")
    keys = vectors[:n_keys]
    queries = vectors[n_keys : n_keys + n_queries]
    return keys, queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose same-item vs cross-item polar codebook behavior.")
    parser.add_argument("--model-name", type=str, default=HF_QWEN_SMALL)
    parser.add_argument("--tasks", nargs="+", default=["qasper", "multifieldqa_en", "2wikimqa"])
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    parser.add_argument("--max-vectors-per-item", type=int, default=4096)
    parser.add_argument("--eval-keys", type=int, default=2048)
    parser.add_argument("--eval-queries", type=int, default=256)
    parser.add_argument("--bits-per-level", type=str, default="4,3,2,2,2,2")
    parser.add_argument("--radius-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lowrank-rank", type=int, default=8)
    parser.add_argument("--lowrank-sparse-k", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "qwen_protocol_diagnosis.json"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    model.eval()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    bits_per_level = [int(part) for part in args.bits_per_level.split(",") if part.strip()]
    quantizer = RecursivePolarQuantizer(dim=head_dim, bits_per_level=bits_per_level, radius_bits=args.radius_bits)
    signs = randomized_hadamard_matrix(head_dim, seed=args.seed)

    items = load_contexts(args.tasks, args.samples_per_task, args.max_words)
    per_item = gather_key_vectors_per_item(
        model,
        tokenizer,
        items,
        max_context_tokens=args.max_context_tokens,
        max_samples_per_item=args.max_vectors_per_item,
    )

    same_item = []
    for item in per_item:
        keys, queries = slice_for_eval(item["vectors"], args.eval_keys, args.eval_queries)
        calibration = item["vectors"]
        same_item.append(
            {
                "task": item["task"],
                "id": item["id"],
                "polar": evaluate_codec(
                    "polar",
                    calibration,
                    keys,
                    queries,
                    quantizer,
                    signs,
                    args.lowrank_rank,
                    args.lowrank_sparse_k,
                ),
                "q4": evaluate_codec(
                    "q4",
                    calibration,
                    keys,
                    queries,
                    quantizer,
                    signs,
                    args.lowrank_rank,
                    args.lowrank_sparse_k,
                ),
                "lowrank_sparse": evaluate_codec(
                    "lowrank_sparse",
                    calibration,
                    keys,
                    queries,
                    quantizer,
                    signs,
                    args.lowrank_rank,
                    args.lowrank_sparse_k,
                ),
            }
        )

    cross_item = []
    if per_item:
        fit_item = per_item[0]
        calibration = fit_item["vectors"]
        for item in per_item[1:]:
            keys, queries = slice_for_eval(item["vectors"], args.eval_keys, args.eval_queries)
            cross_item.append(
                {
                    "fit_task": fit_item["task"],
                    "fit_id": fit_item["id"],
                    "eval_task": item["task"],
                    "eval_id": item["id"],
                    "polar": evaluate_codec(
                        "polar",
                        calibration,
                        keys,
                        queries,
                        quantizer,
                        signs,
                        args.lowrank_rank,
                        args.lowrank_sparse_k,
                    ),
                    "q4": evaluate_codec(
                        "q4",
                        calibration,
                        keys,
                        queries,
                        quantizer,
                        signs,
                        args.lowrank_rank,
                        args.lowrank_sparse_k,
                    ),
                    "lowrank_sparse": evaluate_codec(
                        "lowrank_sparse",
                        calibration,
                        keys,
                        queries,
                        quantizer,
                        signs,
                        args.lowrank_rank,
                        args.lowrank_sparse_k,
                    ),
                }
            )

    result = {
        "model_name": args.model_name,
        "head_dim": head_dim,
        "same_item": same_item,
        "cross_item": cross_item,
        "notes": {
            "current_profiler_protocol": "pseudo-queries are held-out key vectors; keys from all layers are mixed",
            "purpose": "separate same-prompt codebook behavior from cross-prompt generalization before using profiler numbers",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
