"""
SRPQ-v1.1: Spectral Residual Polar Quantization (Corrected)

Fixes from review:
  1. Truncated polar: store intermediate radii at truncation level,
     reconstruct from partial angle stack + boundary radii
  2. Level-1 angle quantization uses circular distance
  3. Background tokens store truncation-level radii for partial reconstruction
  4. Implicit randomized Hadamard (store only sign vector, O(d log d) transform)
  5. psi_perp comes from SVD tail basis Vt[r:].T, not random QR
"""

import numpy as np
import math
import time


def fwht(x: np.ndarray) -> np.ndarray:
    *batch_shape, d = x.shape
    x = x.copy().reshape(-1, d)
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                a = x[:, j].copy()
                b = x[:, j + h].copy()
                x[:, j] = a + b
                x[:, j + h] = a - b
        h *= 2
    x = x / np.sqrt(d)
    return x.reshape(*batch_shape, d)


def randomized_hadamard(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    x_signed = x * signs
    return fwht(x_signed)


def inverse_randomized_hadamard(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    x_h = fwht(x)
    return x_h * signs


def polar_transform_truncated(x: np.ndarray, n_levels: int):
    batch, d = x.shape
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
    l_max = int(math.log2(d))
    l = min(n_levels, l_max)
    angles = []
    r = x.copy()
    for level in range(l):
        r_even = r[:, 0::2]
        r_odd = r[:, 1::2]
        if level == 0:
            psi = np.arctan2(r_odd, r_even) % (2 * np.pi)
        else:
            psi = np.arctan2(np.abs(r_odd), np.abs(r_even) + 1e-30)
        angles.append(psi)
        r = np.sqrt(r_even**2 + r_odd**2 + 1e-30)
    full_radius = np.linalg.norm(x, axis=1)
    return r, angles, full_radius


def inverse_polar_truncated(boundary_radii: np.ndarray, angles):
    r = boundary_radii.copy()
    for level in range(len(angles) - 1, -1, -1):
        psi = angles[level]
        r_cos = r * np.cos(psi)
        r_sin = r * np.sin(psi)
        batch = r_cos.shape[0]
        dim = r_cos.shape[1]
        r = np.zeros((batch, 2 * dim))
        r[:, 0::2] = r_cos
        r[:, 1::2] = r_sin
    return r


def circular_distance(a: np.ndarray, b: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    diff = np.abs(a - b)
    return np.minimum(diff, period - diff)


def build_codebook_circular(samples: np.ndarray, n_centroids: int, period: float = 2 * np.pi, n_iter: int = 50):
    centroids = np.linspace(0, period, n_centroids, endpoint=False)
    centroids += period / (2 * n_centroids)
    for _ in range(n_iter):
        dists = circular_distance(samples[:, np.newaxis], centroids[np.newaxis, :], period)
        assignments = dists.argmin(axis=-1)
        for j in range(n_centroids):
            mask = assignments == j
            if mask.any():
                s = samples[mask]
                mean_sin = np.sin(s * 2 * np.pi / period).mean()
                mean_cos = np.cos(s * 2 * np.pi / period).mean()
                centroids[j] = (np.arctan2(mean_sin, mean_cos) * period / (2 * np.pi)) % period
    return np.sort(centroids)


def build_codebook_linear(samples: np.ndarray, n_centroids: int, n_iter: int = 50):
    s_min, s_max = samples.min(), samples.max()
    centroids = np.linspace(s_min, s_max, n_centroids)
    for _ in range(n_iter):
        dists = np.abs(samples[:, np.newaxis] - centroids[np.newaxis, :])
        assignments = dists.argmin(axis=-1)
        for j in range(n_centroids):
            mask = assignments == j
            if mask.any():
                centroids[j] = samples[mask].mean()
    return np.sort(centroids)


def quantize_to_codebook(values: np.ndarray, codebook: np.ndarray, circular: bool = False, period: float = 2 * np.pi):
    orig_shape = values.shape
    flat = values.reshape(-1)
    if circular:
        dists = circular_distance(flat[:, np.newaxis], codebook[np.newaxis, :], period)
    else:
        dists = np.abs(flat[:, np.newaxis] - codebook[np.newaxis, :])
    indices = dists.argmin(axis=-1).astype(np.uint8)
    return indices.reshape(orig_shape)


def dequantize_from_codebook(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    return codebook[indices.astype(int)]


def quantize_uniform(x, bits, per_channel=True):
    n_levels = 2 ** bits
    if per_channel and x.ndim == 2:
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
    else:
        x_min = x.min()
        x_max = x.max()
    scale = (x_max - x_min) / (n_levels - 1)
    scale = np.maximum(scale, 1e-10)
    zero_point = x_min
    indices = np.round((x - zero_point) / scale).clip(0, n_levels - 1).astype(np.uint8)
    return indices, scale, zero_point


def dequantize_uniform(indices, scale, zero_point):
    return indices.astype(np.float64) * scale + zero_point


class SRPQCompressor:
    def __init__(
        self,
        rank: int = 16,
        n_polar_levels: int = 4,
        coeff_bits: int = 4,
        bit_schedule_innovation=(4, 3, 2, 2),
        bit_schedule_moderate=(3, 2, 2, 2),
        innovation_raw_bits: int = 8,
        tau_h: float = 0.3,
        tau_l: float = 0.1,
    ):
        self.rank = rank
        self.n_levels = n_polar_levels
        self.coeff_bits = coeff_bits
        self.bit_inn = bit_schedule_innovation
        self.bit_mod = bit_schedule_moderate
        self.inn_raw_bits = innovation_raw_bits
        self.tau_h = tau_h
        self.tau_l = tau_l

    def compress(self, K: np.ndarray) -> dict:
        n, d = K.shape
        K = K.astype(np.float64)
        _, S_vals, Vt = np.linalg.svd(K, full_matrices=False)
        r = min(self.rank, len(S_vals))
        psi_r = Vt[:r].T
        psi_perp = Vt[r:].T
        d_prime = d - r
        C = K @ psi_r
        E = K @ psi_perp
        d_pad = 2 ** math.ceil(math.log2(max(d_prime, 2)))
        if d_pad > d_prime:
            E_padded = np.zeros((n, d_pad))
            E_padded[:, :d_prime] = E
        else:
            E_padded = E.copy()
        signs = np.random.choice([-1.0, 1.0], size=d_pad)
        E_precond = randomized_hadamard(E_padded, signs)
        norms_e = np.sum(E**2, axis=1)
        norms_k = np.sum(K**2, axis=1)
        innovation = norms_e / (norms_k + 1e-10)
        inn_mask = innovation > self.tau_h
        mod_mask = (innovation > self.tau_l) & ~inn_mask
        bg_mask = ~inn_mask & ~mod_mask
        inn_idx = np.where(inn_mask)[0]
        mod_idx = np.where(mod_mask)[0]
        bg_idx = np.where(bg_mask)[0]
        c_q, c_scale, c_zero = quantize_uniform(C, self.coeff_bits, per_channel=True)
        boundary_radii, angles, full_radii = polar_transform_truncated(E_precond, self.n_levels)
        codebooks = []
        for l in range(self.n_levels):
            samples = angles[l].ravel()
            max_bits = max(
                self.bit_inn[l] if l < len(self.bit_inn) else 2,
                self.bit_mod[l] if l < len(self.bit_mod) else 2,
            )
            n_centroids = 2 ** max_bits
            if l == 0:
                cb = build_codebook_circular(samples, n_centroids, period=2 * np.pi)
            else:
                cb = build_codebook_linear(samples, n_centroids)
            codebooks.append(cb)
        inn_residual_q = inn_res_scale = inn_res_zero = None
        if len(inn_idx) > 0:
            E_inn = E[inn_idx]
            inn_residual_q, inn_res_scale, inn_res_zero = quantize_uniform(E_inn, self.inn_raw_bits, per_channel=True)
        mod_angle_indices = []
        if len(mod_idx) > 0:
            for l in range(self.n_levels):
                bits_l = self.bit_mod[l] if l < len(self.bit_mod) else 2
                n_cent = 2 ** bits_l
                if l == 0:
                    cb_l = build_codebook_circular(angles[l][mod_idx].ravel(), n_cent, period=2 * np.pi)
                else:
                    cb_l = build_codebook_linear(angles[l][mod_idx].ravel(), n_cent)
                idx = quantize_to_codebook(angles[l][mod_idx], cb_l, circular=(l == 0), period=2 * np.pi)
                mod_angle_indices.append((idx, cb_l))
        bg_boundary_radii_q = bg_br_scale = bg_br_zero = None
        if len(bg_idx) > 0:
            br = boundary_radii[bg_idx]
            bg_boundary_radii_q, bg_br_scale, bg_br_zero = quantize_uniform(br, 4, per_channel=False)
        return {
            "psi_r": psi_r,
            "psi_perp": psi_perp,
            "signs": signs,
            "d_prime": d_prime,
            "d_pad": d_pad,
            "S_vals": S_vals,
            "c_q": c_q,
            "c_scale": c_scale,
            "c_zero": c_zero,
            "innovation": innovation,
            "inn_idx": inn_idx,
            "mod_idx": mod_idx,
            "bg_idx": bg_idx,
            "inn_mask": inn_mask,
            "mod_mask": mod_mask,
            "bg_mask": bg_mask,
            "inn_residual_q": inn_residual_q,
            "inn_res_scale": inn_res_scale,
            "inn_res_zero": inn_res_zero,
            "mod_angle_indices": mod_angle_indices,
            "mod_boundary_radii": boundary_radii[mod_idx] if len(mod_idx) > 0 else None,
            "bg_boundary_radii_q": bg_boundary_radii_q,
            "bg_br_scale": bg_br_scale,
            "bg_br_zero": bg_br_zero,
            "boundary_radii": boundary_radii,
            "angles": angles,
            "codebooks": codebooks,
            "n": n,
            "d": d,
            "r": r,
            "n_levels": self.n_levels,
        }

    def decompress(self, comp: dict, indices: np.ndarray = None) -> np.ndarray:
        n = comp["n"]
        d = comp["d"]
        d_prime = comp["d_prime"]
        d_pad = comp["d_pad"]
        if indices is None:
            indices = np.arange(n)
        n_out = len(indices)
        C_hat = dequantize_uniform(comp["c_q"][indices], comp["c_scale"], comp["c_zero"])
        K_sub = C_hat @ comp["psi_r"].T
        E_hat = np.zeros((n_out, d_prime))
        for local_i, gi in enumerate(indices):
            if comp["inn_mask"][gi]:
                li = np.searchsorted(comp["inn_idx"], gi)
                if li < len(comp["inn_idx"]) and comp["inn_idx"][li] == gi:
                    e = dequantize_uniform(comp["inn_residual_q"][li], comp["inn_res_scale"], comp["inn_res_zero"])
                    E_hat[local_i] = e
            elif comp["mod_mask"][gi]:
                li = np.searchsorted(comp["mod_idx"], gi)
                if li < len(comp["mod_idx"]) and comp["mod_idx"][li] == gi:
                    angles_hat = []
                    for l in range(comp["n_levels"]):
                        if l >= len(comp["mod_angle_indices"]):
                            break
                        idx_arr, cb_l = comp["mod_angle_indices"][l]
                        ang = dequantize_from_codebook(idx_arr[li], cb_l)
                        angles_hat.append(ang[np.newaxis, :])
                    if angles_hat:
                        br = comp["mod_boundary_radii"][li]
                        e_precond = inverse_polar_truncated(br[np.newaxis, :], angles_hat).squeeze(0)
                        e_padded = inverse_randomized_hadamard(e_precond[np.newaxis, :], comp["signs"]).squeeze(0)
                        E_hat[local_i] = e_padded[:d_prime]
            elif comp["bg_mask"][gi]:
                li = np.searchsorted(comp["bg_idx"], gi)
                if li < len(comp["bg_idx"]) and comp["bg_idx"][li] == gi:
                    br_q = comp["bg_boundary_radii_q"][li]
                    br = dequantize_uniform(br_q, comp["bg_br_scale"], comp["bg_br_zero"])
                    default_angles = []
                    dim_at_level = d_pad
                    for l in range(comp["n_levels"]):
                        dim_at_level //= 2
                        da = np.full((1, dim_at_level), np.pi / 4)
                        default_angles.append(da)
                    e_precond = inverse_polar_truncated(br[np.newaxis, :], default_angles).squeeze(0)
                    e_padded = inverse_randomized_hadamard(e_precond[np.newaxis, :], comp["signs"]).squeeze(0)
                    E_hat[local_i] = e_padded[:d_prime]
        K_res = E_hat @ comp["psi_perp"].T
        return K_sub + K_res

    def compute_bits(self, comp: dict) -> dict:
        n = comp["n"]
        d = comp["d"]
        r = comp["r"]
        d_prime = comp["d_prime"]
        d_pad = comp["d_pad"]
        n_inn = len(comp["inn_idx"])
        n_mod = len(comp["mod_idx"])
        n_bg = len(comp["bg_idx"])
        n_boundary = d_pad // (2 ** comp["n_levels"])
        bits_psi_r = d * r * 16
        bits_psi_perp = d * d_prime * 16
        bits_signs = d_pad
        bits_codebooks = sum(len(cb) * 16 for _, cb in comp["mod_angle_indices"]) if comp["mod_angle_indices"] else 0
        bits_overhead = bits_psi_r + bits_psi_perp + bits_signs + bits_codebooks
        bits_coeff = n * r * self.coeff_bits
        bits_coeff_params = r * 16 * 2
        bits_inn = n_inn * d_prime * self.inn_raw_bits if n_inn > 0 else 0
        bits_inn_params = d_prime * 16 * 2 if n_inn > 0 else 0
        bits_mod_angles = 0
        for l in range(comp["n_levels"]):
            if l >= len(comp["mod_angle_indices"]):
                break
            idx_arr, _ = comp["mod_angle_indices"][l]
            bits_l = self.bit_mod[l] if l < len(self.bit_mod) else 2
            n_angles_at_level = idx_arr.shape[1] if idx_arr.ndim > 1 else 1
            bits_mod_angles += n_mod * n_angles_at_level * bits_l
        bits_mod_radii = n_mod * n_boundary * 16
        bits_mod = bits_mod_angles + bits_mod_radii
        bits_bg = n_bg * n_boundary * 4
        bits_bg_params = n_boundary * 16 * 2 if n_bg > 0 else 0
        total = bits_overhead + bits_coeff + bits_coeff_params + bits_inn + bits_inn_params + bits_mod + bits_bg + bits_bg_params
        bpc = total / (n * d)
        return {
            "total_bits": total,
            "bits_per_coord": bpc,
            "compression_ratio": 16.0 / bpc,
            "overhead_bits": bits_overhead,
            "overhead_frac": bits_overhead / total if total > 0 else 0,
            "per_class": {
                "n_inn": n_inn,
                "n_mod": n_mod,
                "n_bg": n_bg,
                "bits_per_inn_token": (bits_inn / n_inn) if n_inn > 0 else 0,
                "bits_per_mod_token": (bits_mod / n_mod) if n_mod > 0 else 0,
                "bits_per_bg_token": (bits_bg / n_bg) if n_bg > 0 else 0,
            },
        }


def make_synthetic_keys(n=2048, d=128, true_rank=12, noise_scale=0.3, seed=42):
    np.random.seed(seed)
    U = np.random.randn(n, true_rank)
    S_true = np.array([10, 8, 6, 5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.8])[:true_rank]
    V = np.linalg.qr(np.random.randn(d, true_rank))[0]
    K = U @ np.diag(S_true) @ V.T + noise_scale * np.random.randn(n, d)
    return K


def test_polar_roundtrip():
    np.random.seed(0)
    x = np.random.randn(32, 64)
    L_full = int(math.log2(64))
    br_full, ang_full, _ = polar_transform_truncated(x, L_full)
    x_hat_full = inverse_polar_truncated(br_full, ang_full)
    err_full = np.max(np.abs(x - x_hat_full))
    br_trunc, ang_trunc, _ = polar_transform_truncated(x, 4)
    x_hat_trunc = inverse_polar_truncated(br_trunc, ang_trunc)
    err_trunc = np.max(np.abs(x - x_hat_trunc))
    return {
        "full_error": err_full,
        "trunc_error": err_trunc,
        "boundary_shape": br_trunc.shape,
    }


def test_hadamard_roundtrip():
    np.random.seed(1)
    d = 64
    x = np.random.randn(16, d)
    signs = np.random.choice([-1.0, 1.0], size=d)
    y = randomized_hadamard(x, signs)
    x_hat = inverse_randomized_hadamard(y, signs)
    err = np.max(np.abs(x - x_hat))
    return err


def test_circular_codebook():
    np.random.seed(2)
    samples = np.concatenate([
        np.random.normal(0.1, 0.05, 100),
        np.random.normal(2 * np.pi - 0.1, 0.05, 100),
        np.random.normal(np.pi, 0.1, 200),
    ]) % (2 * np.pi)
    cb = build_codebook_circular(samples, 4, period=2 * np.pi)
    near_zero = np.any(circular_distance(cb, 0.0, 2 * np.pi) < 0.3)
    near_pi = np.any(circular_distance(cb, np.pi, 2 * np.pi) < 0.3)
    return {"codebook": cb, "near_zero": bool(near_zero), "near_pi": bool(near_pi)}


if __name__ == "__main__":
    print("SRPQ v1.1 candidate")
    print("polar roundtrip", test_polar_roundtrip())
    print("hadamard roundtrip", test_hadamard_roundtrip())
    print("circular codebook", test_circular_codebook())
    K = make_synthetic_keys()
    compressor = SRPQCompressor(rank=16, n_polar_levels=4)
    t0 = time.time()
    comp = compressor.compress(K)
    t_c = time.time() - t0
    t0 = time.time()
    K_hat = compressor.decompress(comp)
    t_d = time.time() - t0
    mse = np.mean((K - K_hat) ** 2)
    rel_mse = mse / np.mean(K ** 2)
    bits = compressor.compute_bits(comp)
    print("rel_mse", rel_mse, "bpc", bits["bits_per_coord"], "ratio", bits["compression_ratio"], "compress", t_c, "decompress", t_d)
