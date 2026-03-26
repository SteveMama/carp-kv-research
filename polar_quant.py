from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def randomized_hadamard_matrix(dim: int, seed: int = 0, device: str | torch.device = "cpu") -> torch.Tensor:
    if not _is_power_of_two(dim):
        raise ValueError(f"Hadamard transform requires a power-of-two dimension, got {dim}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int64)
    signs = signs.to(torch.float32).mul_(2.0).sub_(1.0).to(device)
    return signs


def randomized_hadamard_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != signs.shape[0]:
        raise ValueError(f"Shape mismatch: x has dim {x.shape[-1]}, signs has dim {signs.shape[0]}")
    y = x * signs
    n = y.shape[-1]
    h = 1
    scale = 1.0 / math.sqrt(float(n))
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*y.shape[:-3], -1)
        h *= 2
    return y * scale


def inverse_randomized_hadamard_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != signs.shape[0]:
        raise ValueError(f"Shape mismatch: x has dim {x.shape[-1]}, signs has dim {signs.shape[0]}")
    y = x
    n = y.shape[-1]
    h = 1
    scale = 1.0 / math.sqrt(float(n))
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*y.shape[:-3], -1)
        h *= 2
    return (y * scale) * signs


@dataclass
class PolarCodebooks:
    angle_centroids: list[torch.Tensor]
    radius_centroids: torch.Tensor | None = None
    radius_log_space: bool = False
    radius_eps: float = 1e-6


@dataclass
class QuantizedPolarBatch:
    angle_indices: list[torch.Tensor]
    radius: torch.Tensor


class RecursivePolarQuantizer:
    def __init__(
        self,
        dim: int,
        bits_per_level: list[int],
        radius_bits: int = 8,
        radius_log_space: bool = True,
        radius_eps: float = 1e-6,
    ):
        if not _is_power_of_two(dim):
            raise ValueError(f"Polar recursion requires a power-of-two dimension, got {dim}")
        self.dim = dim
        self.levels = int(math.log2(dim))
        if len(bits_per_level) != self.levels:
            raise ValueError(f"Need exactly {self.levels} bit allocations, got {len(bits_per_level)}")
        self.bits_per_level = bits_per_level
        self.radius_bits = radius_bits
        self.radius_log_space = radius_log_space
        self.radius_eps = radius_eps

    def polar_encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        current = x
        level_angles: list[torch.Tensor] = []
        for _ in range(self.levels):
            paired = current.reshape(*current.shape[:-1], -1, 2)
            radii = torch.linalg.norm(paired, dim=-1)
            angles = torch.atan2(paired[..., 1], paired[..., 0])
            level_angles.append(angles)
            current = radii
        final_radius = current.squeeze(-1)
        return level_angles, final_radius

    def polar_decode(self, level_angles: list[torch.Tensor], radius: torch.Tensor) -> torch.Tensor:
        if len(level_angles) != self.levels:
            raise ValueError(f"Need {self.levels} angle tensors, got {len(level_angles)}")
        current = radius.unsqueeze(-1)
        for angles in reversed(level_angles):
            expanded_radius = current.unsqueeze(-1)
            x = expanded_radius * torch.cos(angles.unsqueeze(-1))
            y = expanded_radius * torch.sin(angles.unsqueeze(-1))
            current = torch.stack([x, y], dim=-1).reshape(*angles.shape[:-1], -1)
        return current

    def fit_codebooks(
        self,
        x: torch.Tensor,
        precondition_signs: torch.Tensor | None = None,
    ) -> PolarCodebooks:
        if precondition_signs is not None:
            x = randomized_hadamard_transform(x, precondition_signs)
        level_angles, radius = self.polar_encode(x)
        angle_centroids: list[torch.Tensor] = []
        for level_idx, (bits, angles) in enumerate(zip(self.bits_per_level, level_angles), start=1):
            k = 2**bits
            flattened = angles.reshape(-1).detach().cpu().numpy()
            centroids = self._fit_angle_kmeans(flattened, k, circular=(level_idx == 1))
            angle_centroids.append(torch.tensor(centroids, dtype=x.dtype, device=x.device))
        radius_centroids = None
        if self.radius_bits > 0:
            radius_values = radius
            if self.radius_log_space:
                radius_values = torch.log(radius_values + self.radius_eps)
            radius_centroids = torch.tensor(
                self._fit_1d_kmeans(radius_values.reshape(-1).detach().cpu().numpy(), 2**self.radius_bits),
                dtype=x.dtype,
                device=x.device,
            )
        return PolarCodebooks(
            angle_centroids=angle_centroids,
            radius_centroids=radius_centroids,
            radius_log_space=self.radius_log_space,
            radius_eps=self.radius_eps,
        )

    def quantize(
        self,
        x: torch.Tensor,
        codebooks: PolarCodebooks,
        precondition_signs: torch.Tensor | None = None,
    ) -> QuantizedPolarBatch:
        conditioned = randomized_hadamard_transform(x, precondition_signs) if precondition_signs is not None else x
        level_angles, radius = self.polar_encode(conditioned)
        angle_indices: list[torch.Tensor] = []
        for level_idx, (angles, centroids) in enumerate(zip(level_angles, codebooks.angle_centroids), start=1):
            if level_idx == 1:
                diff = self._circular_distance(
                    angles.unsqueeze(-1),
                    centroids.view(*([1] * angles.ndim), -1),
                )
            else:
                diff = (angles.unsqueeze(-1) - centroids.view(*([1] * angles.ndim), -1)).abs()
            angle_indices.append(diff.argmin(dim=-1).to(torch.int16))
        if codebooks.radius_centroids is not None:
            radius_values = radius
            if codebooks.radius_log_space:
                radius_values = torch.log(radius_values + codebooks.radius_eps)
            diff = (radius_values.unsqueeze(-1) - codebooks.radius_centroids.view(*([1] * radius.ndim), -1)).abs()
            radius_idx = diff.argmin(dim=-1)
            radius_quant = codebooks.radius_centroids[radius_idx]
            if codebooks.radius_log_space:
                radius_quant = torch.exp(radius_quant) - codebooks.radius_eps
        else:
            radius_quant = radius
        return QuantizedPolarBatch(angle_indices=angle_indices, radius=radius_quant)

    def dequantize(
        self,
        quantized: QuantizedPolarBatch,
        codebooks: PolarCodebooks,
        precondition_signs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        level_angles = [
            centroids[idx.long()]
            for idx, centroids in zip(quantized.angle_indices, codebooks.angle_centroids)
        ]
        reconstructed = self.polar_decode(level_angles, quantized.radius)
        if precondition_signs is not None:
            reconstructed = inverse_randomized_hadamard_transform(reconstructed, precondition_signs)
        return reconstructed

    def angle_statistics(
        self,
        x: torch.Tensor,
        precondition_signs: torch.Tensor | None = None,
    ) -> list[dict[str, float]]:
        if precondition_signs is not None:
            x = randomized_hadamard_transform(x, precondition_signs)
        level_angles, _ = self.polar_encode(x)
        stats: list[dict[str, float]] = []
        for level, angles in enumerate(level_angles, start=1):
            flat = angles.reshape(-1)
            centered = flat - (math.pi / 4.0)
            stats.append(
                {
                    "level": float(level),
                    "mean": float(flat.mean().item()),
                    "std": float(flat.std(unbiased=False).item()),
                    "mean_abs_centered_pi_over_4": float(centered.abs().mean().item()),
                }
            )
        return stats

    @staticmethod
    def _fit_1d_kmeans(values: np.ndarray, k: int, steps: int = 25) -> np.ndarray:
        values = values.astype(np.float32)
        if len(values) == 0:
            return np.zeros(k, dtype=np.float32)
        if len(values) < k:
            padded = np.pad(np.sort(values), (0, k - len(values)), mode="edge")
            return padded.astype(np.float32)
        percentiles = np.linspace(0, 100, k, dtype=np.float32)
        centroids = np.percentile(values, percentiles).astype(np.float32)
        for _ in range(steps):
            distances = np.abs(values[:, None] - centroids[None, :])
            assign = distances.argmin(axis=1)
            new_centroids = centroids.copy()
            for idx in range(k):
                members = values[assign == idx]
                if len(members):
                    new_centroids[idx] = members.mean()
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        return np.sort(centroids)

    @staticmethod
    def _circular_distance(values: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(values - centroids), torch.cos(values - centroids)).abs()

    @staticmethod
    def _fit_angle_kmeans(values: np.ndarray, k: int, circular: bool, steps: int = 25) -> np.ndarray:
        values = values.astype(np.float32)
        if not circular:
            return RecursivePolarQuantizer._fit_1d_kmeans(values, k, steps=steps)
        if len(values) == 0:
            return np.zeros(k, dtype=np.float32)
        if len(values) < k:
            padded = np.pad(np.sort(values), (0, k - len(values)), mode="edge")
            return padded.astype(np.float32)
        percentiles = np.linspace(0, 100, k, dtype=np.float32)
        centroids = np.percentile(values, percentiles).astype(np.float32)
        for _ in range(steps):
            distances = np.abs(np.angle(np.exp(1j * (values[:, None] - centroids[None, :]))))
            assign = distances.argmin(axis=1)
            new_centroids = centroids.copy()
            for idx in range(k):
                members = values[assign == idx]
                if len(members):
                    new_centroids[idx] = np.arctan2(np.sin(members).mean(), np.cos(members).mean())
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        return np.sort(centroids.astype(np.float32))
