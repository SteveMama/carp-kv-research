from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from carp_kv import (
    apply_selector_weights as apply_carp_selector_weights,
    build_topk_labels,
    compute_spectral_innovation,
    evaluate_margin_adaptive_mixture,
    fit_selector_weights as fit_carp_selector_weights,
    selector_feature_matrix as carp_selector_feature_matrix,
)
from main import HF_QWEN_SMALL
from polar_quant import RecursivePolarQuantizer, randomized_hadamard_matrix


def resolve_longbench_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parent / "LongBenchRepo"
    candidates = [
        repo_root / "LongBench" / "data",
        repo_root / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find LongBench data under either "
        f"{repo_root / 'LongBench' / 'data'} or {repo_root / 'data'}. "
        "Clone the LongBench repo into `LongBenchRepo`."
    )


LONG_BENCH_DATA_DIR = resolve_longbench_data_dir()
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_contexts(tasks: list[str], samples_per_task: int, max_words: int) -> list[dict]:
    items: list[dict] = []
    for task in tasks:
        path = LONG_BENCH_DATA_DIR / f"{task}.jsonl"
        chosen = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if raw["length"] > max_words:
                    continue
                chosen.append(
                    {
                        "task": task,
                        "id": raw["_id"],
                        "question": raw["input"],
                        "context": raw["context"],
                        "length": raw["length"],
                    }
                )
                if len(chosen) >= samples_per_task:
                    break
        items.extend(chosen)
    return items


def gather_key_vectors(
    model,
    tokenizer,
    items: list[dict],
    max_context_tokens: int,
    max_samples_per_item: int,
) -> torch.Tensor:
    vectors = []
    for item in items:
        prompt = item["context"]
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_tokens)
        with torch.no_grad():
            outputs = model(**encoded, use_cache=True)
        past_key_values = outputs.past_key_values
        for layer_idx, layer_kv in enumerate(past_key_values):
            key_states = layer_kv[0].detach().cpu()  # [batch, kv_heads, seq, head_dim]
            key_states = key_states.squeeze(0)
            sample = key_states.reshape(-1, key_states.shape[-1])
            if sample.shape[0] > max_samples_per_item:
                sample = sample[:max_samples_per_item]
            vectors.append(sample)
        print(f"profiled {item['task']} len={item['length']} layers={len(past_key_values)}")
    if not vectors:
        raise RuntimeError("No key vectors collected")
    return torch.cat(vectors, dim=0)


def per_channel_q4_reconstruct(calibration: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    max_abs = calibration.abs().amax(dim=0)
    scale = torch.clamp(max_abs / 7.0, min=1e-6)
    q = torch.round(vectors / scale).clamp(-8, 7)
    return q * scale


def lowrank_sparse_reconstruct(calibration: torch.Tensor, vectors: torch.Tensor, rank: int, sparse_k: int) -> torch.Tensor:
    centered_cal = calibration - calibration.mean(dim=0, keepdim=True)
    mean = calibration.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(centered_cal, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    centered_vec = vectors - mean
    coeff = centered_vec @ basis
    lowrank = coeff @ basis.T
    residual = centered_vec - lowrank
    if sparse_k > 0:
        topk = torch.topk(residual.abs(), k=min(sparse_k, residual.shape[-1]), dim=-1)
        sparse = torch.zeros_like(residual)
        sparse.scatter_(1, topk.indices, residual.gather(1, topk.indices))
    else:
        sparse = torch.zeros_like(residual)
    return lowrank + sparse + mean


def evaluate_vectors(
    name: str,
    true_keys: torch.Tensor,
    approx_keys: torch.Tensor,
    queries: torch.Tensor,
    topk: tuple[int, ...] = (1, 8, 16),
) -> dict:
    reconstruction = approx_keys - true_keys
    mse = torch.mean(reconstruction ** 2).item()
    rel_l2 = (
        torch.linalg.norm(reconstruction, dim=-1) /
        torch.clamp(torch.linalg.norm(true_keys, dim=-1), min=1e-6)
    ).mean().item()

    true_scores = queries @ true_keys.T
    approx_scores = queries @ approx_keys.T
    score_rmse = torch.sqrt(torch.mean((true_scores - approx_scores) ** 2)).item()
    denom = torch.clamp(true_scores.abs().mean(), min=1e-6)
    mean_abs_score_err = (true_scores - approx_scores).abs().mean().item()
    rel_score_err = mean_abs_score_err / float(denom.item())

    ordered_true = torch.argsort(true_scores, dim=1, descending=True)
    ordered_approx = torch.argsort(approx_scores, dim=1, descending=True)
    metrics = {
        "name": name,
        "reconstruction_mse": mse,
        "mean_relative_l2": rel_l2,
        "score_rmse": score_rmse,
        "mean_abs_score_error": mean_abs_score_err,
        "mean_relative_score_error": rel_score_err,
    }
    for k in topk:
        truth = ordered_true[:, 0]
        approx_topk = ordered_approx[:, :k]
        contain = (approx_topk == truth.unsqueeze(1)).any(dim=1).float().mean().item()
        exact = (ordered_approx[:, 0] == truth).float().mean().item()
        metrics[f"top{k}_containment"] = contain
        if k == 1:
            metrics["top1_agreement"] = exact
    return metrics


def evaluate_score_mixture(
    name: str,
    true_keys: torch.Tensor,
    queries: torch.Tensor,
    base_scores: torch.Tensor,
    upgraded_scores: torch.Tensor,
    selector_scores: torch.Tensor,
    top_fraction: float,
    topk: tuple[int, ...] = (1, 8, 16),
) -> dict:
    true_scores = queries @ true_keys.T
    num_keys = true_keys.shape[0]
    upgrade_k = max(1, int(round(num_keys * top_fraction)))
    selector_order = torch.argsort(selector_scores, dim=1, descending=True)
    mask = torch.zeros_like(base_scores, dtype=torch.bool)
    mask.scatter_(1, selector_order[:, :upgrade_k], True)
    mixed_scores = torch.where(mask, upgraded_scores, base_scores)

    score_rmse = torch.sqrt(torch.mean((true_scores - mixed_scores) ** 2)).item()
    denom = torch.clamp(true_scores.abs().mean(), min=1e-6)
    mean_abs_score_err = (true_scores - mixed_scores).abs().mean().item()
    rel_score_err = mean_abs_score_err / float(denom.item())

    ordered_true = torch.argsort(true_scores, dim=1, descending=True)
    ordered_mixed = torch.argsort(mixed_scores, dim=1, descending=True)
    metrics = {
        "name": name,
        "top_fraction": top_fraction,
        "score_rmse": score_rmse,
        "mean_abs_score_error": mean_abs_score_err,
        "mean_relative_score_error": rel_score_err,
    }
    for k in topk:
        truth = ordered_true[:, 0]
        approx_topk = ordered_mixed[:, :k]
        contain = (approx_topk == truth.unsqueeze(1)).any(dim=1).float().mean().item()
        exact = (ordered_mixed[:, 0] == truth).float().mean().item()
        metrics[f"top{k}_containment"] = contain
        if k == 1:
            metrics["top1_agreement"] = exact
    return metrics


def bits_per_coord(bits_per_level: list[int], radius_bits: int, dim: int) -> float:
    levels = len(bits_per_level)
    bits_per_vector = sum(
        (2 ** (levels - level_idx)) * bits
        for level_idx, bits in enumerate(bits_per_level, start=1)
    ) + radius_bits
    return bits_per_vector / dim


def row_zscore(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.clip(std, 1e-6, None)


def descending_rank_features(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(scores.shape[1])[None, :]
    return 1.0 - (ranks.astype(np.float32) / max(scores.shape[1] - 1, 1))


def selector_feature_matrix(low_scores: torch.Tensor, lowrank_scores: torch.Tensor) -> np.ndarray:
    low = low_scores.detach().cpu().numpy().astype(np.float32)
    lowrank = lowrank_scores.detach().cpu().numpy().astype(np.float32)
    z_low = row_zscore(low)
    z_lowrank = row_zscore(lowrank)
    delta = z_lowrank - z_low
    abs_delta = np.abs(delta)
    rank_low = descending_rank_features(low)
    rank_lowrank = descending_rank_features(lowrank)
    feat = np.stack(
        [
            z_low,
            z_lowrank,
            delta,
            abs_delta,
            z_low * z_lowrank,
            rank_low,
            rank_lowrank,
            rank_low - rank_lowrank,
        ],
        axis=-1,
    )
    return feat


def fit_selector_weights(
    feature_tensor: np.ndarray,
    labels: np.ndarray,
    steps: int = 400,
    lr: float = 0.1,
    l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    x = feature_tensor.reshape(-1, feature_tensor.shape[-1]).astype(np.float32)
    y = labels.reshape(-1).astype(np.float32)
    positive = float(y.sum())
    negative = float(len(y) - positive)
    if positive < 1:
        raise RuntimeError("No positive selector labels")
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    pos_weight = negative / positive
    sample_weight = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
    for _ in range(steps):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = (probs - y) * sample_weight
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def apply_selector_weights(feature_tensor: np.ndarray, weights: np.ndarray, bias: float) -> torch.Tensor:
    logits = feature_tensor @ weights + bias
    return torch.from_numpy(logits.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Qwen KV angle concentration for PolarQuant-style preconditioning.")
    parser.add_argument("--model-name", type=str, default=HF_QWEN_SMALL)
    parser.add_argument("--tasks", nargs="+", default=["qasper", "multifieldqa_en", "2wikimqa"])
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--max-context-tokens", type=int, default=768)
    parser.add_argument("--max-vectors-per-item", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calibration-vectors", type=int, default=4096)
    parser.add_argument("--eval-keys", type=int, default=2048)
    parser.add_argument("--eval-queries", type=int, default=256)
    parser.add_argument("--selector-train-keys", type=int, default=1024)
    parser.add_argument("--selector-train-queries", type=int, default=128)
    parser.add_argument("--lowrank-rank", type=int, default=8)
    parser.add_argument("--lowrank-sparse-k", type=int, default=4)
    parser.add_argument("--bits-per-level", type=str, default="4,3,2,2,2,2")
    parser.add_argument("--radius-bits", type=int, default=8)
    parser.add_argument("--high-bits-per-level", type=str, default="4,4,3,3,2,2")
    parser.add_argument("--high-radius-bits", type=int, default=8)
    parser.add_argument("--mixed-top-fraction", type=float, default=0.1)
    parser.add_argument("--carp-selector-rank", type=int, default=16)
    parser.add_argument("--carp-positive-k", type=int, default=8)
    parser.add_argument("--carp-base-fraction", type=float, default=0.02)
    parser.add_argument("--carp-max-fraction", type=float, default=0.18)
    parser.add_argument("--carp-pivot-k", type=int, default=8)
    parser.add_argument("--carp-threshold", type=float, default=0.2)
    parser.add_argument("--carp-temperature", type=float, default=0.08)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "qwen_polar_profile.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    model.eval()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    bits_per_level = [int(part) for part in args.bits_per_level.split(",") if part.strip()]
    high_bits_per_level = [int(part) for part in args.high_bits_per_level.split(",") if part.strip()]
    quantizer = RecursivePolarQuantizer(dim=head_dim, bits_per_level=bits_per_level, radius_bits=args.radius_bits)
    high_quantizer = RecursivePolarQuantizer(
        dim=head_dim,
        bits_per_level=high_bits_per_level,
        radius_bits=args.high_radius_bits,
    )
    signs = randomized_hadamard_matrix(head_dim, seed=args.seed)

    items = load_contexts(args.tasks, args.samples_per_task, args.max_words)
    key_vectors = gather_key_vectors(
        model=model,
        tokenizer=tokenizer,
        items=items,
        max_context_tokens=args.max_context_tokens,
        max_samples_per_item=args.max_vectors_per_item,
    ).to(torch.float32)

    raw_stats = quantizer.angle_statistics(key_vectors)
    preconditioned_stats = quantizer.angle_statistics(key_vectors, precondition_signs=signs)

    calibration = key_vectors[: min(len(key_vectors), args.calibration_vectors)]
    codebooks = quantizer.fit_codebooks(calibration, precondition_signs=signs)
    quantized = quantizer.quantize(calibration, codebooks, precondition_signs=signs)
    reconstructed = quantizer.dequantize(quantized, codebooks, precondition_signs=signs)
    mse = torch.mean((calibration - reconstructed) ** 2).item()
    rel_l2 = (
        torch.linalg.norm(calibration - reconstructed, dim=-1) /
        torch.clamp(torch.linalg.norm(calibration, dim=-1), min=1e-6)
    ).mean().item()

    selector_train_start = calibration.shape[0]
    selector_train_end = selector_train_start + args.selector_train_keys + args.selector_train_queries
    selector_train_vectors = key_vectors[selector_train_start:selector_train_end]
    if selector_train_vectors.shape[0] < args.selector_train_keys + args.selector_train_queries:
        raise RuntimeError("Not enough held-out vectors for selector training")
    selector_train_keys = selector_train_vectors[: args.selector_train_keys]
    selector_train_queries = selector_train_vectors[
        args.selector_train_keys: args.selector_train_keys + args.selector_train_queries
    ]

    eval_start = selector_train_end
    eval_end = min(key_vectors.shape[0], eval_start + args.eval_keys + args.eval_queries)
    eval_vectors = key_vectors[eval_start:eval_end]
    if eval_vectors.shape[0] < args.eval_keys + args.eval_queries:
        raise RuntimeError("Not enough held-out vectors for evaluation")
    eval_keys = eval_vectors[: args.eval_keys]
    eval_queries = eval_vectors[args.eval_keys: args.eval_keys + args.eval_queries]

    polar_eval = quantizer.dequantize(
        quantizer.quantize(eval_keys, codebooks, precondition_signs=signs),
        codebooks,
        precondition_signs=signs,
    )
    high_codebooks = high_quantizer.fit_codebooks(calibration, precondition_signs=signs)
    high_polar_eval = high_quantizer.dequantize(
        high_quantizer.quantize(eval_keys, high_codebooks, precondition_signs=signs),
        high_codebooks,
        precondition_signs=signs,
    )
    q4_eval = per_channel_q4_reconstruct(calibration, eval_keys)
    lowrank_eval = lowrank_sparse_reconstruct(
        calibration,
        eval_keys,
        rank=args.lowrank_rank,
        sparse_k=args.lowrank_sparse_k,
    )
    lowrank_train = lowrank_sparse_reconstruct(
        calibration,
        selector_train_keys,
        rank=args.lowrank_rank,
        sparse_k=args.lowrank_sparse_k,
    )

    polar_bits_per_coord = bits_per_coord(bits_per_level, args.radius_bits, quantizer.dim)
    high_polar_bits_per_coord = bits_per_coord(high_bits_per_level, args.high_radius_bits, high_quantizer.dim)

    true_scores = eval_queries @ eval_keys.T
    low_polar_scores = eval_queries @ polar_eval.T
    high_polar_scores = eval_queries @ high_polar_eval.T
    lowrank_selector_scores = eval_queries @ lowrank_eval.T

    selector_true_scores = selector_train_queries @ selector_train_keys.T
    selector_low_polar_scores = selector_train_queries @ quantizer.dequantize(
        quantizer.quantize(selector_train_keys, codebooks, precondition_signs=signs),
        codebooks,
        precondition_signs=signs,
    ).T
    selector_lowrank_scores = selector_train_queries @ lowrank_train.T
    selector_features = selector_feature_matrix(selector_low_polar_scores, selector_lowrank_scores)
    selector_labels = np.zeros(selector_true_scores.shape, dtype=np.float32)
    selector_truth = selector_true_scores.argmax(dim=1).detach().cpu().numpy()
    selector_labels[np.arange(selector_labels.shape[0]), selector_truth] = 1.0
    selector_weights, selector_bias = fit_selector_weights(selector_features, selector_labels)
    eval_selector_features = selector_feature_matrix(low_polar_scores, lowrank_selector_scores)
    learned_selector_scores = apply_selector_weights(eval_selector_features, selector_weights, selector_bias)

    spectral_train = compute_spectral_innovation(calibration, selector_train_keys, rank=args.carp_selector_rank)
    spectral_eval = compute_spectral_innovation(calibration, eval_keys, rank=args.carp_selector_rank)
    carp_train_features = carp_selector_feature_matrix(
        selector_low_polar_scores,
        selector_lowrank_scores,
        spectral_train,
    )
    carp_labels, carp_label_weights = build_topk_labels(
        selector_true_scores,
        positive_k=args.carp_positive_k,
    )
    carp_weights, carp_bias = fit_carp_selector_weights(
        carp_train_features,
        carp_labels,
        carp_label_weights,
    )
    carp_eval_features = carp_selector_feature_matrix(
        low_polar_scores,
        lowrank_selector_scores,
        spectral_eval,
    )
    carp_selector_scores = apply_carp_selector_weights(carp_eval_features, carp_weights, carp_bias)

    comparisons = {
        "polar_quant": evaluate_vectors("polar_quant", eval_keys, polar_eval, eval_queries),
        "polar_quant_high": evaluate_vectors("polar_quant_high", eval_keys, high_polar_eval, eval_queries),
        "q4_per_channel": evaluate_vectors("q4_per_channel", eval_keys, q4_eval, eval_queries),
        "lowrank_sparse": evaluate_vectors("lowrank_sparse", eval_keys, lowrank_eval, eval_queries),
        "mixed_precision_lowrank_selector": evaluate_score_mixture(
            "mixed_precision_lowrank_selector",
            eval_keys,
            eval_queries,
            low_polar_scores,
            high_polar_scores,
            lowrank_selector_scores,
            top_fraction=args.mixed_top_fraction,
        ),
        "mixed_precision_learned_selector": evaluate_score_mixture(
            "mixed_precision_learned_selector",
            eval_keys,
            eval_queries,
            low_polar_scores,
            high_polar_scores,
            learned_selector_scores,
            top_fraction=args.mixed_top_fraction,
        ),
        "mixed_precision_oracle_selector": evaluate_score_mixture(
            "mixed_precision_oracle_selector",
            eval_keys,
            eval_queries,
            low_polar_scores,
            high_polar_scores,
            true_scores,
            top_fraction=args.mixed_top_fraction,
        ),
        "carp_margin_adaptive": evaluate_margin_adaptive_mixture(
            "carp_margin_adaptive",
            eval_keys,
            eval_queries,
            low_polar_scores,
            high_polar_scores,
            carp_selector_scores,
            base_fraction=args.carp_base_fraction,
            max_fraction=args.carp_max_fraction,
            pivot_k=args.carp_pivot_k,
            threshold=args.carp_threshold,
            temperature=args.carp_temperature,
        ),
    }
    comparisons["polar_quant"]["approx_bits_per_coord"] = polar_bits_per_coord
    comparisons["polar_quant_high"]["approx_bits_per_coord"] = high_polar_bits_per_coord
    comparisons["q4_per_channel"]["approx_bits_per_coord"] = 4.0
    comparisons["lowrank_sparse"]["notes"] = {
        "rank": args.lowrank_rank,
        "sparse_k": args.lowrank_sparse_k,
        "compression_proxy": "shared basis + exact sparse residual, unquantized coefficients",
    }
    mixed_bits = (
        (1.0 - args.mixed_top_fraction) * polar_bits_per_coord +
        args.mixed_top_fraction * high_polar_bits_per_coord
    )
    comparisons["mixed_precision_lowrank_selector"]["approx_bits_per_coord"] = mixed_bits
    comparisons["mixed_precision_lowrank_selector"]["base_bits_per_coord"] = polar_bits_per_coord
    comparisons["mixed_precision_lowrank_selector"]["high_bits_per_coord"] = high_polar_bits_per_coord
    comparisons["mixed_precision_learned_selector"]["approx_bits_per_coord"] = mixed_bits
    comparisons["mixed_precision_learned_selector"]["base_bits_per_coord"] = polar_bits_per_coord
    comparisons["mixed_precision_learned_selector"]["high_bits_per_coord"] = high_polar_bits_per_coord
    comparisons["mixed_precision_oracle_selector"]["approx_bits_per_coord"] = mixed_bits
    comparisons["mixed_precision_oracle_selector"]["base_bits_per_coord"] = polar_bits_per_coord
    comparisons["mixed_precision_oracle_selector"]["high_bits_per_coord"] = high_polar_bits_per_coord
    carp_mean_fraction = comparisons["carp_margin_adaptive"]["mean_used_fraction"]
    carp_bits = (
        (1.0 - carp_mean_fraction) * polar_bits_per_coord +
        carp_mean_fraction * high_polar_bits_per_coord
    )
    comparisons["carp_margin_adaptive"]["approx_bits_per_coord"] = carp_bits
    comparisons["carp_margin_adaptive"]["base_bits_per_coord"] = polar_bits_per_coord
    comparisons["carp_margin_adaptive"]["high_bits_per_coord"] = high_polar_bits_per_coord
    comparisons["selector_training"] = {
        "feature_dim": int(selector_features.shape[-1]),
        "train_queries": int(selector_train_queries.shape[0]),
        "train_keys": int(selector_train_keys.shape[0]),
        "weights": selector_weights.tolist(),
        "bias": float(selector_bias),
    }
    comparisons["carp_selector_training"] = {
        "feature_dim": int(carp_train_features.shape[-1]),
        "train_queries": int(selector_train_queries.shape[0]),
        "train_keys": int(selector_train_keys.shape[0]),
        "spectral_rank": int(args.carp_selector_rank),
        "positive_k": int(args.carp_positive_k),
        "weights": carp_weights.tolist(),
        "bias": float(carp_bias),
    }

    result = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "head_dim": head_dim,
        "bits_per_level": bits_per_level,
        "radius_bits": args.radius_bits,
        "high_bits_per_level": high_bits_per_level,
        "high_radius_bits": args.high_radius_bits,
        "mixed_top_fraction": args.mixed_top_fraction,
        "vector_count": int(key_vectors.shape[0]),
        "raw_angle_stats": raw_stats,
        "preconditioned_angle_stats": preconditioned_stats,
        "reconstruction": {
            "mse": mse,
            "mean_relative_l2": rel_l2,
        },
        "comparisons": comparisons,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
