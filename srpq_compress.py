from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from polar_quant import (
    QuantizedPolarBatch,
    RecursivePolarQuantizer,
    PolarCodebooks,
    randomized_hadamard_matrix,
    randomized_hadamard_transform,
    inverse_randomized_hadamard_transform,
)


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def pad_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.shape[-1] == target_dim:
        return x
    if x.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad dim {x.shape[-1]} down to {target_dim}")
    pad = target_dim - x.shape[-1]
    return torch.nn.functional.pad(x, (0, pad))


def compute_complement_basis(basis: torch.Tensor) -> torch.Tensor:
    dim = basis.shape[0]
    if basis.numel() == 0:
        return torch.eye(dim, dtype=basis.dtype, device=basis.device)
    q, _ = torch.linalg.qr(basis, mode="complete")
    return q[:, basis.shape[1]:]


@dataclass
class UniformQuantParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    num_bits: int


@dataclass
class SRPQCompressed:
    psi_r: torch.Tensor
    psi_perp: torch.Tensor
    precond_signs: torch.Tensor
    coeff_quant: torch.Tensor
    coeff_params: UniformQuantParams
    residual_radius: torch.Tensor
    innovation_scores: torch.Tensor
    innovation_mask: torch.Tensor
    moderate_mask: torch.Tensor
    background_mask: torch.Tensor
    innovation_quantized: QuantizedPolarBatch
    innovation_codebooks: PolarCodebooks
    moderate_quantized: QuantizedPolarBatch
    moderate_codebooks: PolarCodebooks
    background_mean_direction: torch.Tensor
    padded_residual_dim: int
    innovation_bits: list[int]
    moderate_bits: list[int]
    coeff_bits: int


class SRPQCompressor:
    def __init__(
        self,
        rank: int = 16,
        innovation_thresholds: tuple[float, float] = (0.3, 0.1),
        innovation_bits: tuple[int, ...] = (4, 2, 2, 2),
        moderate_bits: tuple[int, ...] = (2, 2, 2, 2),
        coeff_bits: int = 4,
        radius_bits: int = 8,
    ) -> None:
        self.rank = rank
        self.tau_h, self.tau_l = innovation_thresholds
        self.innovation_bits = list(innovation_bits)
        self.moderate_bits = list(moderate_bits)
        self.coeff_bits = coeff_bits
        self.radius_bits = radius_bits

    def compress(self, k_pre_rope: torch.Tensor) -> SRPQCompressed:
        if k_pre_rope.ndim != 2:
            raise ValueError(f"Expected (n, d) tensor, got {tuple(k_pre_rope.shape)}")
        x = k_pre_rope.to(torch.float32)
        n, d = x.shape
        rank = min(self.rank, d)

        _, _, vt = torch.linalg.svd(x, full_matrices=False)
        psi_r = vt[:rank].T.contiguous()
        coeff = x @ psi_r
        residual = x - coeff @ psi_r.T

        psi_perp_full = compute_complement_basis(psi_r)
        residual_dim = psi_perp_full.shape[1]
        padded_residual_dim = next_power_of_two(max(1, residual_dim))
        if residual_dim == 0:
            psi_perp = torch.zeros(d, 1, dtype=x.dtype, device=x.device)
            residual_proj = torch.zeros(n, 1, dtype=x.dtype, device=x.device)
        else:
            psi_perp = psi_perp_full
            residual_proj = residual @ psi_perp

        residual_proj_padded = pad_last_dim(residual_proj, padded_residual_dim)
        precond_signs = randomized_hadamard_matrix(padded_residual_dim, seed=7, device=x.device)
        residual_precond = randomized_hadamard_transform(residual_proj_padded, precond_signs)

        coeff_quant, coeff_params = self._quantize_uniform(coeff, self.coeff_bits)

        residual_energy = (residual_proj ** 2).sum(dim=1)
        total_energy = (x ** 2).sum(dim=1).clamp_min(1e-8)
        innovation_scores = residual_energy / total_energy

        innovation_mask = innovation_scores > self.tau_h
        moderate_mask = (innovation_scores > self.tau_l) & ~innovation_mask
        background_mask = ~(innovation_mask | moderate_mask)

        residual_radius = torch.linalg.norm(residual_precond, dim=1)

        innovation_quantizer = self._make_quantizer(padded_residual_dim, self.innovation_bits)
        moderate_quantizer = self._make_quantizer(padded_residual_dim, self.moderate_bits)

        innovation_residuals = residual_precond[innovation_mask]
        moderate_residuals = residual_precond[moderate_mask]

        innovation_codebooks = innovation_quantizer.fit_codebooks(innovation_residuals) if innovation_residuals.shape[0] else self._empty_codebooks(innovation_quantizer, x)
        innovation_quantized = innovation_quantizer.quantize(innovation_residuals, innovation_codebooks) if innovation_residuals.shape[0] else self._empty_quantized(innovation_quantizer, innovation_residuals)

        moderate_codebooks = moderate_quantizer.fit_codebooks(moderate_residuals) if moderate_residuals.shape[0] else self._empty_codebooks(moderate_quantizer, x)
        moderate_quantized = moderate_quantizer.quantize(moderate_residuals, moderate_codebooks) if moderate_residuals.shape[0] else self._empty_quantized(moderate_quantizer, moderate_residuals)

        background_direction = self._compute_background_mean_direction(residual_precond[background_mask], padded_residual_dim, x.device, x.dtype)

        compressed = SRPQCompressed(
            psi_r=psi_r,
            psi_perp=psi_perp,
            precond_signs=precond_signs,
            coeff_quant=coeff_quant,
            coeff_params=coeff_params,
            residual_radius=residual_radius,
            innovation_scores=innovation_scores,
            innovation_mask=innovation_mask,
            moderate_mask=moderate_mask,
            background_mask=background_mask,
            innovation_quantized=innovation_quantized,
            innovation_codebooks=innovation_codebooks,
            moderate_quantized=moderate_quantized,
            moderate_codebooks=moderate_codebooks,
            background_mean_direction=background_direction,
            padded_residual_dim=padded_residual_dim,
            innovation_bits=self.innovation_bits,
            moderate_bits=self.moderate_bits,
            coeff_bits=self.coeff_bits,
        )
        return compressed

    def decompress(self, compressed: SRPQCompressed, indices: torch.Tensor | None = None) -> torch.Tensor:
        coeff = self._dequantize_uniform(compressed.coeff_quant, compressed.coeff_params)
        n = coeff.shape[0]
        if indices is None:
            indices = torch.arange(n, device=coeff.device)
        else:
            indices = indices.to(coeff.device)

        coeff_sel = coeff[indices]
        innovation_mask = compressed.innovation_mask[indices]
        moderate_mask = compressed.moderate_mask[indices]
        background_mask = compressed.background_mask[indices]
        radii = compressed.residual_radius[indices]

        residual_sel = torch.zeros(indices.shape[0], compressed.padded_residual_dim, dtype=coeff.dtype, device=coeff.device)

        if innovation_mask.any():
            innovation_quantizer = self._make_quantizer(compressed.padded_residual_dim, compressed.innovation_bits)
            innovation_full = innovation_quantizer.dequantize(
                compressed.innovation_quantized,
                compressed.innovation_codebooks,
            )
            residual_sel[innovation_mask] = innovation_full[self._subset_positions(compressed.innovation_mask, indices[innovation_mask])]

        if moderate_mask.any():
            moderate_quantizer = self._make_quantizer(compressed.padded_residual_dim, compressed.moderate_bits)
            moderate_full = moderate_quantizer.dequantize(
                compressed.moderate_quantized,
                compressed.moderate_codebooks,
            )
            residual_sel[moderate_mask] = moderate_full[self._subset_positions(compressed.moderate_mask, indices[moderate_mask])]

        if background_mask.any():
            residual_sel[background_mask] = radii[background_mask].unsqueeze(1) * compressed.background_mean_direction.unsqueeze(0)

        residual_sel = inverse_randomized_hadamard_transform(residual_sel, compressed.precond_signs)
        residual_sel = residual_sel[:, :compressed.psi_perp.shape[1]]

        recon = coeff_sel @ compressed.psi_r.T
        if compressed.psi_perp.numel():
            recon = recon + residual_sel @ compressed.psi_perp.T
        return recon

    @staticmethod
    def _subset_positions(mask: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
        full_positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
        lookup = {int(idx.item()): pos for pos, idx in enumerate(full_positions)}
        return torch.tensor([lookup[int(idx.item())] for idx in selected_indices], device=selected_indices.device, dtype=torch.long)

    @staticmethod
    def _compute_background_mean_direction(
        residuals: torch.Tensor,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if residuals.numel() == 0:
            vec = torch.zeros(dim, dtype=dtype, device=device)
            vec[0] = 1.0
            return vec
        mean_vec = residuals.mean(dim=0)
        norm = torch.linalg.norm(mean_vec).clamp_min(1e-8)
        return mean_vec / norm

    @staticmethod
    def _make_quantizer(dim: int, level_bits: list[int]) -> RecursivePolarQuantizer:
        levels = int(math.log2(dim))
        if len(level_bits) > levels:
            raise ValueError(f"Too many polar levels: got {len(level_bits)} for dim {dim}")
        full_bits = level_bits + [0] * (levels - len(level_bits))
        return RecursivePolarQuantizer(dim=dim, bits_per_level=full_bits, radius_bits=8)

    @staticmethod
    def _empty_codebooks(quantizer: RecursivePolarQuantizer, reference: torch.Tensor) -> PolarCodebooks:
        centroids = [torch.zeros(1, dtype=reference.dtype, device=reference.device) for _ in range(quantizer.levels)]
        radius_centroids = torch.zeros(1, dtype=reference.dtype, device=reference.device)
        return PolarCodebooks(angle_centroids=centroids, radius_centroids=radius_centroids)

    @staticmethod
    def _empty_quantized(quantizer: RecursivePolarQuantizer, reference: torch.Tensor) -> QuantizedPolarBatch:
        shape = (0,)
        angle_indices = []
        num_pairs = quantizer.dim // 2
        for level in range(quantizer.levels):
            angle_shape = shape + (max(1, num_pairs),)
            angle_indices.append(torch.zeros(angle_shape, dtype=torch.int16, device=reference.device))
            num_pairs = max(1, num_pairs // 2)
        radius = torch.zeros(shape, dtype=reference.dtype if reference.numel() else torch.float32, device=reference.device)
        return QuantizedPolarBatch(angle_indices=angle_indices, radius=radius)

    @staticmethod
    def _quantize_uniform(x: torch.Tensor, num_bits: int) -> tuple[torch.Tensor, UniformQuantParams]:
        levels = 2 ** num_bits
        x_min = x.amin(dim=0)
        x_max = x.amax(dim=0)
        scale = torch.clamp((x_max - x_min) / max(levels - 1, 1), min=1e-6)
        zero_point = x_min
        q = torch.round((x - zero_point) / scale).clamp(0, levels - 1).to(torch.int16)
        return q, UniformQuantParams(scale=scale, zero_point=zero_point, num_bits=num_bits)

    @staticmethod
    def _dequantize_uniform(q: torch.Tensor, params: UniformQuantParams) -> torch.Tensor:
        return q.to(torch.float32) * params.scale + params.zero_point
