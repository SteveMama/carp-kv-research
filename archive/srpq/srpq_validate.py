from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import HF_QWEN_SMALL
from polar_quant import RecursivePolarQuantizer, randomized_hadamard_matrix
from profile_qwen_kv_polar import load_contexts, gather_key_vectors, evaluate_vectors
from srpq_compress import SRPQCompressor


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def estimate_key_bits_per_token(comp) -> float:
    n, d = comp.coeff_quant.shape
    coeff_bits = d * comp.coeff_bits
    radius_bits = 16
    innovation_levels = comp.innovation_bits + [0] * (int(torch.log2(torch.tensor(comp.padded_residual_dim)).item()) - len(comp.innovation_bits))
    moderate_levels = comp.moderate_bits + [0] * (int(torch.log2(torch.tensor(comp.padded_residual_dim)).item()) - len(comp.moderate_bits))
    innovation_angle_bits = sum((2 ** (len(innovation_levels) - idx)) * bits for idx, bits in enumerate(innovation_levels, start=1))
    moderate_angle_bits = sum((2 ** (len(moderate_levels) - idx)) * bits for idx, bits in enumerate(moderate_levels, start=1))
    token_bits = (
        comp.innovation_mask.float() * (coeff_bits + radius_bits + innovation_angle_bits) +
        comp.moderate_mask.float() * (coeff_bits + radius_bits + moderate_angle_bits) +
        comp.background_mask.float() * (coeff_bits + radius_bits)
    ).mean().item()
    basis_bits = (comp.psi_r.numel() + comp.psi_perp.numel()) * 16 / max(n, 1)
    return token_bits + basis_bits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SRPQ compression against polar baseline on held-out Qwen KV vectors.")
    parser.add_argument("--model-name", type=str, default=HF_QWEN_SMALL)
    parser.add_argument("--tasks", nargs="+", default=["qasper", "multifieldqa_en", "2wikimqa"])
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--max-context-tokens", type=int, default=768)
    parser.add_argument("--max-vectors-per-item", type=int, default=1024)
    parser.add_argument("--calibration-vectors", type=int, default=4096)
    parser.add_argument("--eval-keys", type=int, default=2048)
    parser.add_argument("--eval-queries", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--tau-high", type=float, default=0.3)
    parser.add_argument("--tau-low", type=float, default=0.1)
    parser.add_argument("--coeff-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "srpq_validation.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    model.eval()

    items = load_contexts(args.tasks, args.samples_per_task, args.max_words)
    key_vectors = gather_key_vectors(
        model=model,
        tokenizer=tokenizer,
        items=items,
        max_context_tokens=args.max_context_tokens,
        max_samples_per_item=args.max_vectors_per_item,
    ).to(torch.float32)

    calibration = key_vectors[: min(len(key_vectors), args.calibration_vectors)]
    eval_vectors = key_vectors[calibration.shape[0]: calibration.shape[0] + args.eval_keys + args.eval_queries]
    if eval_vectors.shape[0] < args.eval_keys + args.eval_queries:
        raise RuntimeError("Not enough held-out vectors for evaluation")
    eval_keys = eval_vectors[: args.eval_keys]
    eval_queries = eval_vectors[args.eval_keys: args.eval_keys + args.eval_queries]

    head_dim = eval_keys.shape[1]
    signs = randomized_hadamard_matrix(head_dim, seed=args.seed)
    polar = RecursivePolarQuantizer(dim=head_dim, bits_per_level=[4, 4, 3, 3, 2, 2], radius_bits=8)
    polar_cb = polar.fit_codebooks(calibration, precondition_signs=signs)
    polar_rec = polar.dequantize(polar.quantize(eval_keys, polar_cb, precondition_signs=signs), polar_cb, precondition_signs=signs)
    polar_metrics = evaluate_vectors("polar_quant_high", eval_keys, polar_rec, eval_queries)
    polar_metrics["approx_bits_per_coord"] = (sum((2 ** (polar.levels - idx)) * bits for idx, bits in enumerate(polar.bits_per_level, start=1)) + polar.radius_bits) / head_dim

    srpq = SRPQCompressor(
        rank=args.rank,
        innovation_thresholds=(args.tau_high, args.tau_low),
        coeff_bits=args.coeff_bits,
    )
    srpq_comp = srpq.compress(eval_keys)
    srpq_rec = srpq.decompress(srpq_comp)
    srpq_metrics = evaluate_vectors("srpq", eval_keys, srpq_rec, eval_queries)
    srpq_metrics["estimated_bits_per_key_token"] = estimate_key_bits_per_token(srpq_comp)
    srpq_metrics["innovation_fraction"] = float(srpq_comp.innovation_mask.float().mean().item())
    srpq_metrics["moderate_fraction"] = float(srpq_comp.moderate_mask.float().mean().item())
    srpq_metrics["background_fraction"] = float(srpq_comp.background_mask.float().mean().item())

    result = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "rank": args.rank,
        "thresholds": {"tau_high": args.tau_high, "tau_low": args.tau_low},
        "polar_quant_high": polar_metrics,
        "srpq": srpq_metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
