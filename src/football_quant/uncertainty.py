from __future__ import annotations
import numpy as np


def summarize_distribution(values) -> dict:
    """Summarize an array of bootstrap replicate values (e.g. per-replicate
    probabilities) into a central estimate + dispersion, honestly reflecting
    how much the estimate moves across plausible re-fits of the model."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("No hay réplicas de bootstrap para resumir.")
    q = np.percentile(values, [5, 10, 50, 90, 95])
    return {
        "mean": float(values.mean()),
        "P05": float(q[0]),
        "P10": float(q[1]),
        "P50": float(q[2]),
        "P90": float(q[3]),
        "P95": float(q[4]),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def fragility(base_p: float, stressed_p: float) -> dict:
    """Separate downside (stress hurts this selection) from upside (stress
    helps it), instead of collapsing both into a single number that erases
    the sign. A market where stress *improves* the probability is very
    different from one where stress destroys it, even if a naive
    max(base - stress, 0) would report 0 for both."""
    diff = float(base_p - stressed_p)
    return {
        "downside_risk": max(diff, 0.0),
        "upside_potential": max(-diff, 0.0),
    }


def data_quality_score(n_matches: int, freshness_score: float = 1.0, split_score: float = 1.0, source_score: float = 1.0) -> float:
    quantity = min(n_matches / 30.0, 1.0)
    return float(np.clip(0.40 * quantity + 0.25 * freshness_score + 0.20 * split_score + 0.15 * source_score, 0, 1))