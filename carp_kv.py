from __future__ import annotations

import math

import numpy as np
import torch


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


def compute_spectral_innovation(
    calibration: torch.Tensor,
    vectors: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    centered_cal = calibration - calibration.mean(dim=0, keepdim=True)
    mean = calibration.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered_cal, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    centered = vectors - mean
    coeff = centered @ basis
    lowrank = coeff @ basis.T
    residual = centered - lowrank
    residual_energy = (residual ** 2).sum(dim=-1)
    total_energy = (centered ** 2).sum(dim=-1).clamp_min(1e-8)
    innovation = residual_energy / total_energy
    return innovation.to(torch.float32)


def selector_feature_matrix(
    low_scores: torch.Tensor,
    lowrank_scores: torch.Tensor,
    spectral_innovation: torch.Tensor,
) -> np.ndarray:
    low = low_scores.detach().cpu().numpy().astype(np.float32)
    lowrank = lowrank_scores.detach().cpu().numpy().astype(np.float32)
    innovation = spectral_innovation.detach().cpu().numpy().astype(np.float32)
    innovation = innovation[None, :]
    innovation_z = (innovation - innovation.mean()) / max(float(innovation.std()), 1e-6)

    z_low = row_zscore(low)
    z_lowrank = row_zscore(lowrank)
    delta = z_lowrank - z_low
    abs_delta = np.abs(delta)
    rank_low = descending_rank_features(low)
    rank_lowrank = descending_rank_features(lowrank)
    low_margin_anchor = np.partition(low, kth=max(low.shape[1] - 8, 0), axis=1)[:, -8][:, None]
    relative_to_anchor = z_low - row_zscore(low_margin_anchor.repeat(low.shape[1], axis=1))

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
            innovation_z.repeat(low.shape[0], axis=0),
            z_low * innovation_z,
            delta * innovation_z,
            abs_delta * innovation_z,
            relative_to_anchor,
        ],
        axis=-1,
    )
    return feat.astype(np.float32)


def build_topk_labels(
    true_scores: torch.Tensor,
    positive_k: int,
    top1_weight: float = 2.0,
    other_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    order = torch.argsort(true_scores, dim=1, descending=True)
    labels = np.zeros(true_scores.shape, dtype=np.float32)
    weights = np.ones(true_scores.shape, dtype=np.float32)
    topk = order[:, :positive_k].detach().cpu().numpy()
    row_idx = np.arange(labels.shape[0])[:, None]
    labels[row_idx, topk] = 1.0
    weights[row_idx, topk] = other_weight
    top1 = order[:, 0].detach().cpu().numpy()
    weights[np.arange(labels.shape[0]), top1] = top1_weight
    return labels, weights


def fit_selector_weights(
    feature_tensor: np.ndarray,
    labels: np.ndarray,
    label_weights: np.ndarray,
    steps: int = 500,
    lr: float = 0.08,
    l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    x = feature_tensor.reshape(-1, feature_tensor.shape[-1]).astype(np.float32)
    y = labels.reshape(-1).astype(np.float32)
    w_user = label_weights.reshape(-1).astype(np.float32)
    positive = float(y.sum())
    negative = float(len(y) - positive)
    if positive < 1:
        raise RuntimeError("No positive selector labels")
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    pos_weight = negative / positive
    sample_weight = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32) * w_user
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


def evaluate_margin_adaptive_mixture(
    name: str,
    true_keys: torch.Tensor,
    queries: torch.Tensor,
    low_scores: torch.Tensor,
    high_scores: torch.Tensor,
    selector_scores: torch.Tensor,
    base_fraction: float,
    max_fraction: float,
    pivot_k: int = 8,
    threshold: float = 0.2,
    temperature: float = 0.08,
    topk: tuple[int, ...] = (1, 8, 16),
) -> dict:
    true_scores = queries @ true_keys.T
    num_keys = true_keys.shape[0]
    max_budget = max(1, int(round(num_keys * max_fraction)))
    base_budget = max(1, int(round(num_keys * base_fraction)))

    selector_np = selector_scores.detach().cpu().numpy()
    low_np = low_scores.detach().cpu().numpy()
    high_np = high_scores.detach().cpu().numpy()
    mixed = low_np.copy()

    used_counts: list[int] = []
    ambiguities: list[float] = []
    for row in range(low_np.shape[0]):
        ordered_low = np.sort(low_np[row])[::-1]
        pivot = min(max(1, pivot_k), len(ordered_low) - 1)
        margin = float(ordered_low[0] - ordered_low[pivot])
        logit = float((threshold - margin) / max(temperature, 1e-6))
        logit = max(min(logit, 30.0), -30.0)
        ambiguity = 1.0 / (1.0 + math.exp(-logit))
        budget = base_budget + int(round((max_budget - base_budget) * ambiguity))
        budget = max(1, min(num_keys, budget))
        used_counts.append(budget)
        ambiguities.append(ambiguity)
        chosen = np.argpartition(-selector_np[row], budget - 1)[:budget]
        mixed[row, chosen] = high_np[row, chosen]

    mixed_scores = torch.from_numpy(mixed.astype(np.float32))
    score_rmse = torch.sqrt(torch.mean((true_scores - mixed_scores) ** 2)).item()
    denom = torch.clamp(true_scores.abs().mean(), min=1e-6)
    mean_abs_score_err = (true_scores - mixed_scores).abs().mean().item()
    rel_score_err = mean_abs_score_err / float(denom.item())

    ordered_true = torch.argsort(true_scores, dim=1, descending=True)
    ordered_mixed = torch.argsort(mixed_scores, dim=1, descending=True)
    metrics = {
        "name": name,
        "base_fraction": base_fraction,
        "max_fraction": max_fraction,
        "mean_used_fraction": float(np.mean(used_counts) / num_keys),
        "mean_ambiguity": float(np.mean(ambiguities)),
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


def margin_entropy_risk(
    scores: torch.Tensor,
    tau: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D scores, got {tuple(scores.shape)}")
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    norm_scores = (scores - mean) / std
    sorted_scores = torch.sort(norm_scores, dim=1, descending=True).values
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    probs = torch.softmax(norm_scores, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1)
    entropy = entropy / math.log(scores.shape[1] + 1)
    risk_logit = (-margin / max(tau, 1e-6)) + lam * entropy
    risk = torch.sigmoid(risk_logit)
    return risk, margin, entropy


def promoted_fraction_from_risk(
    risk: torch.Tensor,
    base_fraction: float,
    max_fraction: float,
    gamma: float = 1.0,
) -> torch.Tensor:
    risk = risk.clamp(0.0, 1.0)
    scaled = risk.pow(gamma)
    return base_fraction + (max_fraction - base_fraction) * scaled
