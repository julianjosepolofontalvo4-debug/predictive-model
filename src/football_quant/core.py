from __future__ import annotations
import numpy as np
from scipy.stats import nbinom, poisson

EPS = 1e-9

def temporal_weights(n: int, decay: float = 0.08) -> np.ndarray:
    """Return normalized weights for rows ordered oldest to newest."""
    if n <= 0:
        return np.array([], dtype=float)
    age = np.arange(n - 1, -1, -1, dtype=float)
    w = np.exp(-decay * age)
    return w / w.sum()

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        raise ValueError("No hay datos para estimar la media.")
    return float(np.average(values, weights=weights))

def shrink_mean(observed: float, n_effective: float, prior: float, k: float = 5.0) -> float:
    n_effective = max(float(n_effective), 0.0)
    k = max(float(k), EPS)
    return float((n_effective * observed + k * prior) / (n_effective + k))

def weighted_variance(values: np.ndarray, weights: np.ndarray, mean: float | None = None) -> float:
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = weighted_mean(values, weights)
    return float(np.average((values - mean) ** 2, weights=weights))

def fit_count_distribution(values, weights, prior_mean: float, shrink_k: float = 5.0, effective_n_override: float | None = None) -> dict:
    values = np.asarray(values, dtype=float)
    mean_obs = weighted_mean(values, weights)
    effective_n = 1.0 / np.sum(weights ** 2) if effective_n_override is None else max(float(effective_n_override), 0.0)
    mean = shrink_mean(mean_obs, effective_n, prior_mean, shrink_k)
    var = weighted_variance(values, weights, mean_obs)
    disp = var / max(mean_obs, EPS)

    if len(values) < 8 or disp <= 1.10:
        model, dispersion = "poisson", 1.0
    else:
        model, dispersion = "negative_binomial", max(disp, 1.10)

    return {
        "observed_mean": float(mean_obs),
        "mean": float(mean),
        "variance": float(var),
        "dispersion_index": float(dispersion),
        "effective_n": float(effective_n),
        "model": model,
    }

def sample_count(model: str, mean: float, dispersion: float, rng: np.random.Generator) -> int:
    mean = max(float(mean), EPS)
    if model == "poisson":
        return int(rng.poisson(mean))
    variance = max(mean * dispersion, mean * 1.000001)
    r = mean * mean / max(variance - mean, EPS)
    p = r / (r + mean)
    return int(rng.negative_binomial(r, p))

def pmf(model: str, x: int, mean: float, dispersion: float = 1.0) -> float:
    mean = max(float(mean), EPS)
    if model == "poisson":
        return float(poisson.pmf(x, mean))
    variance = max(mean * dispersion, mean * 1.000001)
    r = mean * mean / max(variance - mean, EPS)
    p = r / (r + mean)
    return float(nbinom.pmf(x, r, p))
