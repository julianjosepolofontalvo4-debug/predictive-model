from __future__ import annotations
import numpy as np
from .core import sample_count

def simulate_match(params: dict, n: int = 100_000, seed: int | None = None, shocks: dict | None = None):
    rng = np.random.default_rng(seed)
    shocks = shocks or {}
    samples = {}

    for key, p in params.items():
        mean = max(p["mean"] * float(shocks.get(key, 1.0)), 1e-9)
        samples[key] = np.fromiter(
            (sample_count(p["model"], mean, p["dispersion"], rng) for _ in range(n)),
            dtype=np.int64,
            count=n,
        )
    return samples

def quantiles(values):
    q = np.percentile(values, [5, 10, 50, 90, 95])
    return {"P05": float(q[0]), "P10": float(q[1]), "P50": float(q[2]), "P90": float(q[3]), "P95": float(q[4])}
