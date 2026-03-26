from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from carp_cache_eval import CARPCache, build_prompt, patch_qwen_attention, restore_qwen_attention
from longbench_subset_eval import DEFAULT_TASKS, load_longbench_items, load_prompt_config, make_question_prompt, truncate_middle_by_tokens
from main import HF_QWEN_SMALL, RESULTS_DIR


RESULT_PATH = RESULTS_DIR / "carp_multistep_eval.json"


def decode_steps(
    model,
    tokenizer,
    prompt: str,
    max_context_tokens: int,
    decode_steps: int,
    carp_cfg: dict | None,
) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_tokens)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if carp_cfg is None:
        cache = DynamicCache(config=model.config)
    else:
        cache = CARPCache(
            num_layers=model.config.num_hidden_layers,
            selector_weights=np.asarray(carp_cfg["weights"], dtype=np.float32),
            selector_bias=float(carp_cfg["bias"]),
            low_bits=carp_cfg["low_bits"],
            high_bits=carp_cfg["high_bits"],
            base_fraction=float(carp_cfg["base_fraction"]),
            max_fraction=float(carp_cfg["max_fraction"]),
            risk_tau=float(carp_cfg["risk_tau"]),
            risk_lambda=float(carp_cfg["risk_lambda"]),
            risk_gamma=float(carp_cfg.get("risk_gamma", 1.0)),
            risk_beta=float(carp_cfg.get("risk_beta", 0.0)),
            anchor_ratio=float(carp_cfg.get("anchor_ratio", 0.25)),
            exact_fraction=float(carp_cfg.get("exact_fraction", 0.0)),
            exact_head_risk_threshold=float(carp_cfg.get("exact_head_risk_threshold", 2.0)),
        )

    generated_ids: list[int] = []
    step_stats: list[dict[str, float]] = []
    step_logits: list[np.ndarray] = []
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        first_logits = outputs.logits[:, -1, :].detach().to(torch.float32).cpu().numpy()[0]
        step_logits.append(first_logits)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(int(next_token.item()))
        if carp_cfg is not None and hasattr(cache, "collect_stats"):
            stats = cache.collect_stats()
            step_stats.append({"step": 1, **stats})

        for step in range(2, decode_steps + 1):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
                dim=1,
            )
            cache_pos = torch.tensor([attention_mask.shape[1] - 1], dtype=torch.long)
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos,
            )
            step_logits.append(outputs.logits[:, -1, :].detach().to(torch.float32).cpu().numpy()[0])
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(int(next_token.item()))
            if carp_cfg is not None and hasattr(cache, "collect_stats"):
                stats = cache.collect_stats()
                step_stats.append({"step": step, **stats})

    return {
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "step_stats": step_stats,
        "step_logits": step_logits,
    }


def normalized_entropy_from_logits(logits: np.ndarray) -> float:
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    probs /= probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return entropy / float(np.log(len(probs)))


def decode_steps_entropy_fallback(
    model,
    tokenizer,
    prompt: str,
    max_context_tokens: int,
    decode_steps: int,
    carp_cfg: dict,
    entropy_threshold: float,
) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_tokens)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    exact_cache = DynamicCache(config=model.config)
    carp_cache = CARPCache(
        num_layers=model.config.num_hidden_layers,
        selector_weights=np.asarray(carp_cfg["weights"], dtype=np.float32),
        selector_bias=float(carp_cfg["bias"]),
        low_bits=carp_cfg["low_bits"],
        high_bits=carp_cfg["high_bits"],
        base_fraction=float(carp_cfg["base_fraction"]),
        max_fraction=float(carp_cfg["max_fraction"]),
        risk_tau=float(carp_cfg["risk_tau"]),
        risk_lambda=float(carp_cfg["risk_lambda"]),
        risk_gamma=float(carp_cfg.get("risk_gamma", 1.0)),
        risk_beta=float(carp_cfg.get("risk_beta", 0.0)),
        anchor_ratio=float(carp_cfg.get("anchor_ratio", 0.25)),
        exact_fraction=float(carp_cfg.get("exact_fraction", 0.0)),
        exact_head_risk_threshold=float(carp_cfg.get("exact_head_risk_threshold", 2.0)),
    )

    generated_ids: list[int] = []
    fallback_steps: list[bool] = []
    carp_entropies: list[float] = []
    step_stats: list[dict[str, float]] = []
    hybrid_logits: list[np.ndarray] = []
    with torch.no_grad():
        exact_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=exact_cache,
            use_cache=True,
        )
        carp_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=carp_cache,
            use_cache=True,
        )
        exact_logits = exact_outputs.logits[:, -1, :].detach().to(torch.float32).cpu().numpy()[0]
        carp_logits = carp_outputs.logits[:, -1, :].detach().to(torch.float32).cpu().numpy()[0]
        entropy = normalized_entropy_from_logits(carp_logits)
        use_exact = entropy > entropy_threshold
        chosen_logits = exact_logits if use_exact else carp_logits
        hybrid_logits.append(chosen_logits)
        carp_entropies.append(entropy)
        fallback_steps.append(use_exact)
        next_token = torch.tensor([[int(np.argmax(chosen_logits))]], dtype=input_ids.dtype)
        generated_ids.append(int(next_token.item()))
        if hasattr(carp_cache, "collect_stats"):
            stats = carp_cache.collect_stats()
            step_stats.append({"step": 1, "used_exact_step": float(use_exact), "carp_entropy": entropy, **stats})

        for step in range(2, decode_steps + 1):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
                dim=1,
            )
            cache_pos = torch.tensor([attention_mask.shape[1] - 1], dtype=torch.long)
            exact_outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=exact_cache,
                use_cache=True,
                cache_position=cache_pos,
            )
            carp_outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=carp_cache,
                use_cache=True,
                cache_position=cache_pos,
            )
            exact_logits = exact_outputs.logits[:, -1, :].detach().to(torch.float32).cpu().numpy()[0]
            carp_logits = carp_outputs.logits[:, -1, :].detach().to(torch.float32).cpu().numpy()[0]
            entropy = normalized_entropy_from_logits(carp_logits)
            use_exact = entropy > entropy_threshold
            chosen_logits = exact_logits if use_exact else carp_logits
            hybrid_logits.append(chosen_logits)
            carp_entropies.append(entropy)
            fallback_steps.append(use_exact)
            next_token = torch.tensor([[int(np.argmax(chosen_logits))]], dtype=input_ids.dtype)
            generated_ids.append(int(next_token.item()))
            if hasattr(carp_cache, "collect_stats"):
                stats = carp_cache.collect_stats()
                step_stats.append({"step": step, "used_exact_step": float(use_exact), "carp_entropy": entropy, **stats})

    return {
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "step_stats": step_stats,
        "step_logits": hybrid_logits,
        "fallback_steps": fallback_steps,
        "carp_entropies": carp_entropies,
    }


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    return limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-step exact vs CARP generation comparison.")
    parser.add_argument("--model-name", type=str, default=HF_QWEN_SMALL)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-length-words", type=int, default=3000)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--risk-tau", type=float, default=2.0)
    parser.add_argument("--risk-lambda", type=float, default=0.5)
    parser.add_argument("--risk-gamma", type=float, default=1.0)
    parser.add_argument("--risk-beta", type=float, default=1.0)
    parser.add_argument("--anchor-ratio", type=float, default=0.25)
    parser.add_argument("--exact-fraction", type=float, default=0.0)
    parser.add_argument("--exact-head-risk-threshold", type=float, default=0.7)
    parser.add_argument("--entropy-fallback-threshold", type=float, default=-1.0)
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
        "max_fraction": 0.15,
        "risk_tau": args.risk_tau,
        "risk_lambda": args.risk_lambda,
        "risk_gamma": args.risk_gamma,
        "risk_beta": args.risk_beta,
        "anchor_ratio": args.anchor_ratio,
        "exact_fraction": args.exact_fraction,
        "exact_head_risk_threshold": args.exact_head_risk_threshold,
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
                context, _, _ = truncate_middle_by_tokens(tokenizer, item.context, args.max_context_tokens)
                prompt = build_prompt(context, question_prompt)

                exact = decode_steps(
                    model,
                    tokenizer,
                    prompt,
                    args.max_context_tokens,
                    args.decode_steps,
                    carp_cfg=None,
                )
                carp = decode_steps(
                    model,
                    tokenizer,
                    prompt,
                    args.max_context_tokens,
                    args.decode_steps,
                    carp_cfg=carp_cfg,
                )
                hybrid = None
                if args.entropy_fallback_threshold >= 0.0:
                    hybrid = decode_steps_entropy_fallback(
                        model,
                        tokenizer,
                        prompt,
                        args.max_context_tokens,
                        args.decode_steps,
                        carp_cfg=carp_cfg,
                        entropy_threshold=args.entropy_fallback_threshold,
                    )
                exact_ids = exact["generated_ids"]
                carp_ids = carp["generated_ids"]
                prefix = longest_common_prefix(exact_ids, carp_ids)
                match_rate = float(np.mean([int(a == b) for a, b in zip(exact_ids, carp_ids)]))
                overlap = len(set(exact_ids) & set(carp_ids)) / max(len(set(exact_ids)), 1)
                mean_exact_heads = float(
                    np.mean([s.get("mean_exact_head_fraction", 0.0) for s in carp["step_stats"]])
                ) if carp["step_stats"] else 0.0
                step_kls: list[float] = []
                for exact_logits, carp_logits in zip(exact["step_logits"], carp["step_logits"]):
                    exact_probs = np.exp(exact_logits - np.max(exact_logits))
                    exact_probs /= exact_probs.sum()
                    carp_probs = np.exp(carp_logits - np.max(carp_logits))
                    carp_probs /= carp_probs.sum()
                    step_kls.append(float(np.sum(exact_probs * (np.log(exact_probs + 1e-12) - np.log(carp_probs + 1e-12)))))
                records.append(
                    {
                        "task": task,
                        "item_id": item.item_id,
                        "exact_text": exact["generated_text"],
                        "carp_text": carp["generated_text"],
                        "exact_ids": exact_ids,
                        "carp_ids": carp_ids,
                        "step_kls": step_kls,
                        "first_divergence_step": prefix + 1 if prefix < args.decode_steps else None,
                        "token_match_rate": match_rate,
                        "set_overlap": overlap,
                        "mean_exact_head_fraction": mean_exact_heads,
                        "hybrid": None,
                    }
                )
                if hybrid is not None:
                    hybrid_ids = hybrid["generated_ids"]
                    hybrid_prefix = longest_common_prefix(exact_ids, hybrid_ids)
                    hybrid_match_rate = float(np.mean([int(a == b) for a, b in zip(exact_ids, hybrid_ids)]))
                    hybrid_overlap = len(set(exact_ids) & set(hybrid_ids)) / max(len(set(exact_ids)), 1)
                    hybrid_step_kls: list[float] = []
                    for exact_logits, hybrid_logits in zip(exact["step_logits"], hybrid["step_logits"]):
                        exact_probs = np.exp(exact_logits - np.max(exact_logits))
                        exact_probs /= exact_probs.sum()
                        hybrid_probs = np.exp(hybrid_logits - np.max(hybrid_logits))
                        hybrid_probs /= hybrid_probs.sum()
                        hybrid_step_kls.append(float(np.sum(exact_probs * (np.log(exact_probs + 1e-12) - np.log(hybrid_probs + 1e-12)))))
                    records[-1]["hybrid"] = {
                        "text": hybrid["generated_text"],
                        "ids": hybrid_ids,
                        "step_kls": hybrid_step_kls,
                        "fallback_steps": hybrid["fallback_steps"],
                        "carp_entropies": hybrid["carp_entropies"],
                        "first_divergence_step": hybrid_prefix + 1 if hybrid_prefix < args.decode_steps else None,
                        "token_match_rate": hybrid_match_rate,
                        "set_overlap": hybrid_overlap,
                        "fallback_rate": float(np.mean([1.0 if x else 0.0 for x in hybrid["fallback_steps"]])),
                    }
    finally:
        if patched:
            restore_qwen_attention(original_forward)

    summary: dict[str, dict[str, float]] = {}
    for task in args.tasks:
        rows = [r for r in records if r["task"] == task]
        if not rows:
            continue
        divergence = [r["first_divergence_step"] or (args.decode_steps + 1) for r in rows]
        summary[task] = {
            "token_match_rate": float(np.mean([r["token_match_rate"] for r in rows])),
            "first_divergence_step": float(np.mean(divergence)),
            "set_overlap": float(np.mean([r["set_overlap"] for r in rows])),
            "mean_exact_head_fraction": float(np.mean([r["mean_exact_head_fraction"] for r in rows])),
            "mean_final_step_kl": float(np.mean([r["step_kls"][-1] for r in rows])),
        }
        hybrid_rows = [r["hybrid"] for r in rows if r.get("hybrid") is not None]
        if hybrid_rows:
            hybrid_divergence = [r["first_divergence_step"] or (args.decode_steps + 1) for r in hybrid_rows]
            summary[task]["hybrid_token_match_rate"] = float(np.mean([r["token_match_rate"] for r in hybrid_rows]))
            summary[task]["hybrid_first_divergence_step"] = float(np.mean(hybrid_divergence))
            summary[task]["hybrid_set_overlap"] = float(np.mean([r["set_overlap"] for r in hybrid_rows]))
            summary[task]["hybrid_mean_final_step_kl"] = float(np.mean([r["step_kls"][-1] for r in hybrid_rows]))
            summary[task]["hybrid_fallback_rate"] = float(np.mean([r["fallback_rate"] for r in hybrid_rows]))

    result = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "decode_steps": args.decode_steps,
        "carp": carp_cfg,
        "records": records,
        "summary": summary,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
