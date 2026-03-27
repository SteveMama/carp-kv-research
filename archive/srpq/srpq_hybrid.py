from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def quantize_uniform(x: np.ndarray, bits: int, per_channel: bool = True):
    n_levels = 2 ** bits
    if per_channel and x.ndim == 2:
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
    else:
        x_min = x.min()
        x_max = x.max()
    scale = (x_max - x_min) / max(n_levels - 1, 1)
    scale = np.maximum(scale, 1e-10)
    zero_point = x_min
    indices = np.round((x - zero_point) / scale).clip(0, n_levels - 1).astype(np.uint8)
    return indices, scale, zero_point


def dequantize_uniform(indices: np.ndarray, scale: np.ndarray, zero_point: np.ndarray):
    return indices.astype(np.float64) * scale + zero_point


def quantize_symmetric_per_row(values: np.ndarray, bits: int):
    levels = 2 ** bits
    qmax = levels // 2 - 1
    max_abs = np.max(np.abs(values), axis=1, keepdims=True)
    scale = np.maximum(max_abs / max(qmax, 1), 1e-10)
    q = np.round(values / scale).clip(-qmax - 1, qmax).astype(np.int16)
    return q, scale.squeeze(1)


def dequantize_symmetric_per_row(q: np.ndarray, scale: np.ndarray):
    return q.astype(np.float64) * scale[:, np.newaxis]


@dataclass
class SRPQHybridCompressed:
    psi_r: np.ndarray
    psi_perp: np.ndarray
    coeff_q: np.ndarray
    coeff_scale: np.ndarray
    coeff_zero: np.ndarray
    innovation_scores: np.ndarray
    innovation_idx: np.ndarray
    moderate_idx: np.ndarray
    background_idx: np.ndarray
    innovation_res_q: np.ndarray | None
    innovation_res_scale: np.ndarray | None
    innovation_res_zero: np.ndarray | None
    moderate_shell_idx: np.ndarray | None
    moderate_shell_q: np.ndarray | None
    moderate_shell_scale: np.ndarray | None
    rank: int
    coeff_bits: int
    innovation_bits: int
    shell_k: int
    shell_bits: int
    n: int
    d: int
    d_prime: int


class SRPQHybridCompressor:
    def __init__(
        self,
        rank: int = 16,
        coeff_bits: int = 4,
        innovation_bits: int = 8,
        shell_k: int = 8,
        shell_bits: int = 6,
        tau_h: float = 0.45,
        tau_l: float = 0.18,
    ) -> None:
        self.rank = rank
        self.coeff_bits = coeff_bits
        self.innovation_bits = innovation_bits
        self.shell_k = shell_k
        self.shell_bits = shell_bits
        self.tau_h = tau_h
        self.tau_l = tau_l

    def compress(self, K: np.ndarray) -> SRPQHybridCompressed:
        K = K.astype(np.float64)
        n, d = K.shape
        _, S_vals, Vt = np.linalg.svd(K, full_matrices=False)
        rank = min(self.rank, len(S_vals))
        psi_r = Vt[:rank].T
        psi_perp = Vt[rank:].T
        d_prime = d - rank

        coeff = K @ psi_r
        residual = K @ psi_perp

        coeff_q, coeff_scale, coeff_zero = quantize_uniform(coeff, self.coeff_bits, per_channel=True)

        residual_energy = np.sum(residual ** 2, axis=1)
        total_energy = np.sum(K ** 2, axis=1) + 1e-10
        innovation_scores = residual_energy / total_energy

        innovation_mask = innovation_scores > self.tau_h
        moderate_mask = (innovation_scores > self.tau_l) & ~innovation_mask
        background_mask = ~(innovation_mask | moderate_mask)

        innovation_idx = np.where(innovation_mask)[0]
        moderate_idx = np.where(moderate_mask)[0]
        background_idx = np.where(background_mask)[0]

        innovation_res_q = innovation_res_scale = innovation_res_zero = None
        if innovation_idx.size:
            innovation_res = residual[innovation_idx]
            innovation_res_q, innovation_res_scale, innovation_res_zero = quantize_uniform(
                innovation_res,
                self.innovation_bits,
                per_channel=True,
            )

        moderate_shell_idx = moderate_shell_q = moderate_shell_scale = None
        if moderate_idx.size:
            moderate_res = residual[moderate_idx]
            k = min(self.shell_k, d_prime)
            topk = np.argpartition(np.abs(moderate_res), -k, axis=1)[:, -k:]
            gathered = np.take_along_axis(moderate_res, topk, axis=1)
            shell_q, shell_scale = quantize_symmetric_per_row(gathered, self.shell_bits)
            order = np.argsort(topk, axis=1)
            moderate_shell_idx = np.take_along_axis(topk, order, axis=1).astype(np.uint8)
            moderate_shell_q = np.take_along_axis(shell_q, order, axis=1)
            moderate_shell_scale = shell_scale

        return SRPQHybridCompressed(
            psi_r=psi_r,
            psi_perp=psi_perp,
            coeff_q=coeff_q,
            coeff_scale=coeff_scale,
            coeff_zero=coeff_zero,
            innovation_scores=innovation_scores,
            innovation_idx=innovation_idx,
            moderate_idx=moderate_idx,
            background_idx=background_idx,
            innovation_res_q=innovation_res_q,
            innovation_res_scale=innovation_res_scale,
            innovation_res_zero=innovation_res_zero,
            moderate_shell_idx=moderate_shell_idx,
            moderate_shell_q=moderate_shell_q,
            moderate_shell_scale=moderate_shell_scale,
            rank=rank,
            coeff_bits=self.coeff_bits,
            innovation_bits=self.innovation_bits,
            shell_k=self.shell_k,
            shell_bits=self.shell_bits,
            n=n,
            d=d,
            d_prime=d_prime,
        )

    def decompress(self, comp: SRPQHybridCompressed, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            indices = np.arange(comp.n)
        coeff = dequantize_uniform(comp.coeff_q[indices], comp.coeff_scale, comp.coeff_zero)
        K_sub = coeff @ comp.psi_r.T
        E_hat = np.zeros((len(indices), comp.d_prime), dtype=np.float64)

        innovation_lookup = {int(idx): pos for pos, idx in enumerate(comp.innovation_idx)}
        moderate_lookup = {int(idx): pos for pos, idx in enumerate(comp.moderate_idx)}

        for out_pos, global_idx in enumerate(indices):
            gi = int(global_idx)
            if gi in innovation_lookup:
                pos = innovation_lookup[gi]
                E_hat[out_pos] = dequantize_uniform(
                    comp.innovation_res_q[pos],
                    comp.innovation_res_scale,
                    comp.innovation_res_zero,
                )
            elif gi in moderate_lookup:
                pos = moderate_lookup[gi]
                shell_vals = dequantize_symmetric_per_row(
                    comp.moderate_shell_q[pos:pos + 1],
                    comp.moderate_shell_scale[pos:pos + 1],
                )[0]
                shell_idx = comp.moderate_shell_idx[pos].astype(np.int64)
                E_hat[out_pos, shell_idx] = shell_vals

        K_res = E_hat @ comp.psi_perp.T if comp.d_prime else np.zeros_like(K_sub)
        return K_sub + K_res

    def compute_bits(self, comp: SRPQHybridCompressed) -> dict:
        n = comp.n
        d = comp.d
        r = comp.rank
        d_prime = comp.d_prime
        n_inn = len(comp.innovation_idx)
        n_mod = len(comp.moderate_idx)
        n_bg = len(comp.background_idx)

        bits_psi_r = d * r * 16
        bits_psi_perp = d * d_prime * 16
        bits_overhead = bits_psi_r + bits_psi_perp

        bits_coeff = n * r * comp.coeff_bits
        bits_coeff_params = r * 16 * 2

        bits_inn = n_inn * d_prime * comp.innovation_bits
        bits_inn_params = d_prime * 16 * 2 if n_inn > 0 else 0

        index_bits = max(1, math.ceil(math.log2(max(d_prime, 2))))
        bits_mod = n_mod * comp.shell_k * (comp.shell_bits + index_bits)
        bits_mod_scale = n_mod * 16

        total = bits_overhead + bits_coeff + bits_coeff_params + bits_inn + bits_inn_params + bits_mod + bits_mod_scale
        bpc = total / (n * d)
        return {
            "total_bits": total,
            "bits_per_coord": bpc,
            "compression_ratio": 16.0 / bpc,
            "overhead_frac": bits_overhead / total if total else 0.0,
            "per_class": {
                "n_inn": n_inn,
                "n_mod": n_mod,
                "n_bg": n_bg,
                "frac_inn": n_inn / n,
                "frac_mod": n_mod / n,
                "frac_bg": n_bg / n,
            },
        }
