from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import Cache, DynamicLayer
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from carp_kv import (
    apply_selector_weights,
    compute_spectral_innovation,
    margin_entropy_risk,
    promoted_fraction_from_risk,
    selector_feature_matrix,
)
from longbench_subset_eval import DEFAULT_TASKS, load_longbench_items, load_prompt_config, make_question_prompt, truncate_middle_by_tokens
from main import HF_QWEN_SMALL, RESULTS_DIR
from polar_quant import RecursivePolarQuantizer, randomized_hadamard_matrix
from profile_qwen_kv_polar import lowrank_sparse_reconstruct


RESULT_PATH = RESULTS_DIR / "carp_cache_eval.json"


def build_prompt(context: str, question_prompt: str) -> str:
    return (
        "Using only the provided conversation context, answer with only the exact short answer. "
        "Do not explain.\n\n"
        f"Context:\n{context}\n\n"
        f"{question_prompt}\nAnswer:"
    )


def group_query_heads(query_states: torch.Tensor, kv_heads: int) -> torch.Tensor:
    batch, q_heads, q_len, head_dim = query_states.shape
    if q_heads == kv_heads:
        return query_states
    if q_heads % kv_heads != 0:
        return query_states[:, :kv_heads]
    group = q_heads // kv_heads
    return query_states.view(batch, kv_heads, group, q_len, head_dim).mean(dim=2)


class CARPLayer(DynamicLayer):
    def __init__(
        self,
        layer_idx: int,
        selector_weights: np.ndarray,
        selector_bias: float,
        low_bits: list[int],
        high_bits: list[int],
        base_fraction: float,
        max_fraction: float,
        base_codec: str = "polar",
        upgrade_codec: str = "high_polar",
        selector_mode: str = "learned",
        risk_tau: float = 1.0,
        risk_lambda: float = 0.0,
        risk_gamma: float = 1.0,
        risk_beta: float = 0.0,
        anchor_ratio: float = 0.25,
        exact_fraction: float = 0.0,
        exact_head_risk_threshold: float = 2.0,
        spectral_rank: int = 16,
        lowrank_rank: int = 8,
        lowrank_sparse_k: int = 4,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.selector_weights = selector_weights
        self.selector_bias = selector_bias
        self.low_bits = low_bits
        self.high_bits = high_bits
        self.base_fraction = base_fraction
        self.max_fraction = max_fraction
        self.base_codec = base_codec
        self.upgrade_codec = upgrade_codec
        self.selector_mode = selector_mode
        self.risk_tau = risk_tau
        self.risk_lambda = risk_lambda
        self.risk_gamma = risk_gamma
        self.risk_beta = risk_beta
        self.anchor_ratio = anchor_ratio
        self.exact_fraction = exact_fraction
        self.exact_head_risk_threshold = exact_head_risk_threshold
        self.spectral_rank = spectral_rank
        self.lowrank_rank = lowrank_rank
        self.lowrank_sparse_k = lowrank_sparse_k
        self.seed = seed
        self.last_budget_fractions: list[float] = []
        self.last_risks: list[float] = []
        self.last_margins: list[float] = []
        self.last_entropies: list[float] = []
        self.last_disagreements: list[float] = []
        self.last_exact_fractions: list[float] = []
        self.last_exact_heads: list[float] = []

    @staticmethod
    def _row_zscore_torch(scores: torch.Tensor) -> torch.Tensor:
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (scores - mean) / std

    def _heuristic_selector_scores(
        self,
        low_scores: torch.Tensor,
        lowrank_scores: torch.Tensor,
        innovation: torch.Tensor,
    ) -> torch.Tensor:
        z_low = self._row_zscore_torch(low_scores)
        z_lowrank = self._row_zscore_torch(lowrank_scores)
        innovation_z = (innovation - innovation.mean()) / innovation.std().clamp_min(1e-6)
        return (0.65 * z_low[0]) + (0.25 * z_lowrank[0]) + (0.10 * innovation_z)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_budget_fractions = []
        self.last_risks = []
        self.last_margins = []
        self.last_entropies = []
        self.last_disagreements = []
        self.last_exact_fractions = []
        self.last_exact_heads = []
        keys, values = super().update(key_states, value_states, cache_kwargs)
        if cache_kwargs is None:
            return keys, values
        query_states = cache_kwargs.get("query_states")
        if query_states is None:
            return keys, values
        if keys.shape[-2] < 16:
            return keys, values
        if query_states.shape[-2] != 1:
            return keys, values
        if keys.shape[0] != 1:
            return keys, values

        q_grouped = group_query_heads(query_states, keys.shape[1])
        mixed_keys = torch.empty_like(keys)

        for head_idx in range(keys.shape[1]):
            head_keys = keys[0, head_idx].to(torch.float32)
            head_query = q_grouped[0, head_idx, 0].to(torch.float32)
            exact_scores = head_query @ head_keys.T
            head_dim = head_keys.shape[-1]
            signs = randomized_hadamard_matrix(
                head_dim,
                seed=self.seed + self.layer_idx * 31 + head_idx,
                device=head_keys.device,
            )
            low_q = RecursivePolarQuantizer(dim=head_dim, bits_per_level=self.low_bits, radius_bits=8)
            high_q = RecursivePolarQuantizer(dim=head_dim, bits_per_level=self.high_bits, radius_bits=8)

            if self.base_codec == "polar":
                low_codebooks = low_q.fit_codebooks(head_keys, precondition_signs=signs)
                low_recon = low_q.dequantize(
                    low_q.quantize(head_keys, low_codebooks, precondition_signs=signs),
                    low_codebooks,
                    precondition_signs=signs,
                )
            elif self.base_codec == "q4":
                low_recon = per_channel_q4_reconstruct(head_keys, head_keys)
            else:
                raise ValueError(f"Unsupported base codec: {self.base_codec}")

            if self.upgrade_codec == "high_polar":
                high_codebooks = high_q.fit_codebooks(head_keys, precondition_signs=signs)
                high_recon = high_q.dequantize(
                    high_q.quantize(head_keys, high_codebooks, precondition_signs=signs),
                    high_codebooks,
                    precondition_signs=signs,
                )
            elif self.upgrade_codec == "exact":
                high_recon = head_keys
            else:
                raise ValueError(f"Unsupported upgrade codec: {self.upgrade_codec}")

            lowrank_recon = lowrank_sparse_reconstruct(
                head_keys,
                head_keys,
                rank=min(self.lowrank_rank, head_keys.shape[-1] - 1),
                sparse_k=min(self.lowrank_sparse_k, head_keys.shape[-1]),
            )
            low_scores = (head_query @ low_recon.T).unsqueeze(0)
            lowrank_scores = (head_query @ lowrank_recon.T).unsqueeze(0)
            innovation = compute_spectral_innovation(
                head_keys,
                head_keys,
                rank=min(self.spectral_rank, head_keys.shape[-1] - 1),
            )
            if self.selector_mode == "learned":
                feat = selector_feature_matrix(low_scores, lowrank_scores, innovation)
                selector_scores = apply_selector_weights(feat, self.selector_weights, self.selector_bias)[0]
            elif self.selector_mode == "heuristic":
                selector_scores = self._heuristic_selector_scores(low_scores, lowrank_scores, innovation)
            else:
                raise ValueError(f"Unsupported selector mode: {self.selector_mode}")

            risk, margin, entropy = margin_entropy_risk(low_scores, tau=self.risk_tau, lam=self.risk_lambda)
            topk_for_disagreement = min(8, head_keys.shape[0])
            low_top = torch.topk(low_scores[0], k=topk_for_disagreement).indices
            lowrank_top = torch.topk(lowrank_scores[0], k=topk_for_disagreement).indices
            overlap = np.intersect1d(
                low_top.detach().cpu().numpy(),
                lowrank_top.detach().cpu().numpy(),
                assume_unique=False,
            ).size
            disagreement = 1.0 - (overlap / max(topk_for_disagreement, 1))
            if self.risk_beta != 0.0:
                risk_logit = torch.logit(risk.clamp(1e-6, 1 - 1e-6)) + (self.risk_beta * disagreement)
                risk = torch.sigmoid(risk_logit)
            if float(risk[0].item()) >= self.exact_head_risk_threshold:
                mixed_keys[0, head_idx] = head_keys.to(keys.dtype)
                self.last_budget_fractions.append(1.0)
                self.last_risks.append(float(risk[0].item()))
                self.last_margins.append(float(margin[0].item()))
                self.last_entropies.append(float(entropy[0].item()))
                self.last_disagreements.append(float(disagreement))
                self.last_exact_fractions.append(1.0)
                self.last_exact_heads.append(1.0)
                continue
            budget_fraction = float(
                promoted_fraction_from_risk(
                    risk,
                    self.base_fraction,
                    self.max_fraction,
                    gamma=self.risk_gamma,
                )[0].item()
            )
            budget = max(1, int(round(head_keys.shape[0] * budget_fraction)))
            anchor_budget = min(budget, max(1, int(round(budget * self.anchor_ratio))))
            anchor_idx = torch.topk(low_scores[0], k=anchor_budget).indices
            remaining_budget = max(0, budget - anchor_budget)
            if remaining_budget > 0:
                masked_selector = selector_scores.clone()
                masked_selector[anchor_idx] = -float("inf")
                selector_idx = torch.topk(masked_selector, k=remaining_budget).indices
                promote = torch.unique(torch.cat([anchor_idx, selector_idx], dim=0))
            else:
                promote = anchor_idx
            mixed_head = low_recon.clone()
            mixed_head[promote] = high_recon[promote]
            exact_budget = min(
                promote.numel(),
                max(0, int(round(head_keys.shape[0] * self.exact_fraction))),
            )
            if exact_budget > 0:
                high_scores = head_query @ high_recon.T
                score_delta = (exact_scores - high_scores).abs()
                exact_order = torch.argsort(exact_scores, descending=True)
                exact_rank = torch.empty_like(exact_order, dtype=torch.float32)
                exact_rank[exact_order] = torch.arange(
                    exact_scores.numel(),
                    device=exact_scores.device,
                    dtype=torch.float32,
                )
                exact_rank = 1.0 - (exact_rank / max(exact_scores.numel() - 1, 1))
                centered_keys = head_keys - head_keys.mean(dim=0, keepdim=True)
                channel_std = centered_keys.std(dim=0, keepdim=True).clamp_min(1e-6)
                channel_z = (centered_keys / channel_std).abs()
                top_channels = min(4, channel_z.shape[1])
                outlier_score = torch.topk(channel_z, k=top_channels, dim=1).values.mean(dim=1)
                outlier_score = outlier_score / outlier_score.mean().clamp_min(1e-6)
                critical_metric = (
                    score_delta[promote]
                    * (1.0 + exact_rank[promote])
                    * (1.0 + 0.5 * outlier_score[promote])
                )
                exact_local = torch.topk(critical_metric, k=exact_budget).indices
                exact_idx = promote[exact_local]
                mixed_head[exact_idx] = head_keys[exact_idx]
            mixed_keys[0, head_idx] = mixed_head.to(keys.dtype)
            self.last_budget_fractions.append(budget_fraction)
            self.last_risks.append(float(risk[0].item()))
            self.last_margins.append(float(margin[0].item()))
            self.last_entropies.append(float(entropy[0].item()))
            self.last_disagreements.append(float(disagreement))
            self.last_exact_fractions.append(exact_budget / max(head_keys.shape[0], 1))
            self.last_exact_heads.append(0.0)

        return mixed_keys, values


class CARPCache(Cache):
    def __init__(
        self,
        num_layers: int,
        selector_weights: np.ndarray,
        selector_bias: float,
        low_bits: list[int],
        high_bits: list[int],
        base_fraction: float,
        max_fraction: float,
        base_codec: str,
        upgrade_codec: str,
        selector_mode: str,
        risk_tau: float,
        risk_lambda: float,
        risk_gamma: float,
        risk_beta: float,
        anchor_ratio: float,
        exact_fraction: float,
        exact_head_risk_threshold: float,
    ) -> None:
        layers = [
            CARPLayer(
                layer_idx=i,
                selector_weights=selector_weights,
                selector_bias=selector_bias,
                low_bits=low_bits,
                high_bits=high_bits,
                base_fraction=base_fraction,
                max_fraction=max_fraction,
                base_codec=base_codec,
                upgrade_codec=upgrade_codec,
                selector_mode=selector_mode,
                risk_tau=risk_tau,
                risk_lambda=risk_lambda,
                risk_gamma=risk_gamma,
                risk_beta=risk_beta,
                anchor_ratio=anchor_ratio,
                exact_fraction=exact_fraction,
                exact_head_risk_threshold=exact_head_risk_threshold,
            )
            for i in range(num_layers)
        ]
        super().__init__(layers=layers)

    def collect_stats(self) -> dict[str, float]:
        fractions: list[float] = []
        risks: list[float] = []
        margins: list[float] = []
        entropies: list[float] = []
        disagreements: list[float] = []
        exact_fractions: list[float] = []
        exact_heads: list[float] = []
        for layer in self.layers:
            fractions.extend(getattr(layer, "last_budget_fractions", []))
            risks.extend(getattr(layer, "last_risks", []))
            margins.extend(getattr(layer, "last_margins", []))
            entropies.extend(getattr(layer, "last_entropies", []))
            disagreements.extend(getattr(layer, "last_disagreements", []))
            exact_fractions.extend(getattr(layer, "last_exact_fractions", []))
            exact_heads.extend(getattr(layer, "last_exact_heads", []))
        def _mean(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0
        def _max(xs: list[float]) -> float:
            return float(np.max(xs)) if xs else 0.0
        return {
            "mean_budget_fraction": _mean(fractions),
            "max_budget_fraction": _max(fractions),
            "mean_risk": _mean(risks),
            "max_risk": _max(risks),
            "mean_margin": _mean(margins),
            "min_margin": float(np.min(margins)) if margins else 0.0,
            "mean_entropy": _mean(entropies),
            "max_entropy": _max(entropies),
            "mean_disagreement": _mean(disagreements),
            "max_disagreement": _max(disagreements),
            "mean_exact_fraction": _mean(exact_fractions),
            "max_exact_fraction": _max(exact_fractions),
            "mean_exact_head_fraction": _mean(exact_heads),
        }


def patch_qwen_attention() -> tuple[object, bool]:
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
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
    return original, True


def restore_qwen_attention(original) -> None:
    Qwen2Attention.forward = original


def second_step_decode(model, tokenizer, prompt: str, use_carp: bool, carp_cfg: dict) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=carp_cfg["max_context_tokens"])
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if use_carp:
        cache = CARPCache(
            num_layers=model.config.num_hidden_layers,
            selector_weights=np.asarray(carp_cfg["weights"], dtype=np.float32),
            selector_bias=float(carp_cfg["bias"]),
            low_bits=carp_cfg["low_bits"],
            high_bits=carp_cfg["high_bits"],
            base_fraction=float(carp_cfg["base_fraction"]),
            max_fraction=float(carp_cfg["max_fraction"]),
            base_codec=carp_cfg.get("base_codec", "polar"),
            upgrade_codec=carp_cfg.get("upgrade_codec", "high_polar"),
            selector_mode=carp_cfg.get("selector_mode", "learned"),
            risk_tau=float(carp_cfg["risk_tau"]),
            risk_lambda=float(carp_cfg["risk_lambda"]),
            risk_gamma=float(carp_cfg.get("risk_gamma", 1.0)),
            risk_beta=float(carp_cfg.get("risk_beta", 0.0)),
            anchor_ratio=float(carp_cfg.get("anchor_ratio", 0.25)),
            exact_fraction=float(carp_cfg.get("exact_fraction", 0.0)),
            exact_head_risk_threshold=float(carp_cfg.get("exact_head_risk_threshold", 2.0)),
        )
    else:
        cache = DynamicCache(config=model.config)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        first_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
            dim=1,
        )
        cache_pos = torch.tensor([attention_mask.shape[1] - 1], dtype=torch.long)
        outputs = model(
            input_ids=first_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos,
        )
        second_logits = outputs.logits[:, -1, :].detach().to(torch.float32).cpu()
        second_top1 = int(torch.argmax(second_logits, dim=-1).item())
        second_top5 = torch.topk(second_logits, k=5, dim=-1).indices[0].tolist()
        cache_stats = cache.collect_stats() if use_carp and hasattr(cache, "collect_stats") else None

    result = {
        "first_token_id": int(first_token.item()),
        "first_token_text": tokenizer.decode([int(first_token.item())], skip_special_tokens=True),
        "second_top1_id": second_top1,
        "second_top1_text": tokenizer.decode([second_top1], skip_special_tokens=True),
        "second_top5_ids": second_top5,
        "second_top5_text": [tokenizer.decode([tok], skip_special_tokens=True) for tok in second_top5],
        "second_logits": second_logits.numpy(),
    }
    if cache_stats is not None:
        result["cache_stats"] = cache_stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end CARP cache evaluation on a tiny LongBench subset.")
    parser.add_argument("--model-name", type=str, default=HF_QWEN_SMALL)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-length-words", type=int, default=3000)
    parser.add_argument("--full-max-context-tokens", type=int, default=768)
    parser.add_argument("--carp-max-fraction", type=float, default=0.15)
    parser.add_argument("--base-codec", type=str, choices=["polar", "q4"], default="polar")
    parser.add_argument("--upgrade-codec", type=str, choices=["high_polar", "exact"], default="high_polar")
    parser.add_argument("--selector-mode", type=str, choices=["learned", "heuristic"], default="learned")
    parser.add_argument("--risk-tau", type=float, default=1.0)
    parser.add_argument("--risk-lambda", type=float, default=0.0)
    parser.add_argument("--risk-gamma", type=float, default=1.0)
    parser.add_argument("--risk-beta", type=float, default=0.0)
    parser.add_argument("--anchor-ratio", type=float, default=0.25)
    parser.add_argument("--exact-fraction", type=float, default=0.0)
    parser.add_argument("--exact-head-risk-threshold", type=float, default=2.0)
    parser.add_argument("--output", type=str, default=str(RESULT_PATH))
    args = parser.parse_args()

    profile = json.loads((RESULTS_DIR / "qwen_polar_profile.json").read_text(encoding="utf-8"))
    carp_info = profile["comparisons"]["carp_selector_training"]
    carp_cfg = {
        "weights": carp_info["weights"],
        "bias": carp_info["bias"],
        "low_bits": profile["bits_per_level"],
        "high_bits": profile["high_bits_per_level"],
        "base_fraction": profile["comparisons"]["carp_margin_adaptive"]["mean_used_fraction"],
        "max_fraction": args.carp_max_fraction,
        "base_codec": args.base_codec,
        "upgrade_codec": args.upgrade_codec,
        "selector_mode": args.selector_mode,
        "risk_tau": args.risk_tau,
        "risk_lambda": args.risk_lambda,
        "risk_gamma": args.risk_gamma,
        "risk_beta": args.risk_beta,
        "anchor_ratio": args.anchor_ratio,
        "exact_fraction": args.exact_fraction,
        "exact_head_risk_threshold": args.exact_head_risk_threshold,
        "max_context_tokens": args.full_max_context_tokens,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    model.eval()

    prompt_map, _ = load_prompt_config()
    original_forward, patched = patch_qwen_attention()

    try:
        records: list[dict] = []
        for task in args.tasks:
            items = load_longbench_items(task, args.samples_per_task, args.max_length_words)
            for item in items:
                question_prompt = make_question_prompt(task, item.input, prompt_map[task])
                context, _, _ = truncate_middle_by_tokens(tokenizer, item.context, args.full_max_context_tokens)
                prompt = build_prompt(context, question_prompt)

                base_out = second_step_decode(model, tokenizer, prompt, use_carp=False, carp_cfg=carp_cfg)
                carp_out = second_step_decode(model, tokenizer, prompt, use_carp=True, carp_cfg=carp_cfg)

                base_logits = np.asarray(base_out.pop("second_logits"), dtype=np.float64)
                carp_logits = np.asarray(carp_out.pop("second_logits"), dtype=np.float64)
                base_probs = np.exp(base_logits - base_logits.max(axis=-1, keepdims=True))
                base_probs /= base_probs.sum(axis=-1, keepdims=True)
                carp_probs = np.exp(carp_logits - carp_logits.max(axis=-1, keepdims=True))
                carp_probs /= carp_probs.sum(axis=-1, keepdims=True)
                kl = float(np.sum(base_probs * (np.log(base_probs + 1e-12) - np.log(carp_probs + 1e-12))))

                records.append(
                    {
                        "task": task,
                        "item_id": item.item_id,
                        "baseline": base_out,
                        "carp": carp_out,
                        "first_token_match": base_out["first_token_id"] == carp_out["first_token_id"],
                        "second_top1_match": base_out["second_top1_id"] == carp_out["second_top1_id"],
                        "second_top1_in_carp_top5": base_out["second_top1_id"] in carp_out["second_top5_ids"],
                        "second_step_kl": kl,
                    }
                )
    finally:
        if patched:
            restore_qwen_attention(original_forward)

    summary: dict[str, dict[str, float]] = {}
    for task in args.tasks:
        task_rows = [r for r in records if r["task"] == task]
        if not task_rows:
            continue
        summary[task] = {
            "first_token_match": float(np.mean([float(r["first_token_match"]) for r in task_rows])),
            "second_top1_match": float(np.mean([float(r["second_top1_match"]) for r in task_rows])),
            "second_top1_in_carp_top5": float(np.mean([float(r["second_top1_in_carp_top5"]) for r in task_rows])),
            "second_step_kl": float(np.mean([r["second_step_kl"] for r in task_rows])),
        }

    result = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "carp": {
            "base_fraction": carp_cfg["base_fraction"],
            "max_fraction": carp_cfg["max_fraction"],
            "base_codec": carp_cfg["base_codec"],
            "upgrade_codec": carp_cfg["upgrade_codec"],
            "selector_mode": carp_cfg["selector_mode"],
            "risk_tau": carp_cfg["risk_tau"],
            "risk_lambda": carp_cfg["risk_lambda"],
            "risk_gamma": carp_cfg["risk_gamma"],
            "risk_beta": carp_cfg["risk_beta"],
            "anchor_ratio": carp_cfg["anchor_ratio"],
            "exact_fraction": carp_cfg["exact_fraction"],
            "exact_head_risk_threshold": carp_cfg["exact_head_risk_threshold"],
            "low_bits": carp_cfg["low_bits"],
            "high_bits": carp_cfg["high_bits"],
        },
        "records": records,
        "summary": summary,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
