from __future__ import annotations
import numpy as np
from .uncertainty import summarize_distribution, fragility

# Thresholds used to tell "trivial" (near-certain either way) lines apart
# from lines that are actually useful for decision-making.
TRIVIAL_HIGH = 0.90
TRIVIAL_LOW = 0.10


def build_market_row(name, line, side, boot_probs, boot_probs_stressed):
    """Build one market row from bootstrap-of-the-dataset replicate arrays
    (see engine._bootstrap_market_distributions). boot_probs / boot_probs_stressed
    are arrays with one entry per bootstrap replicate: the probability of this
    market resolving True under that replicate's re-fitted model. This is what
    lets P05/P95 reflect real parameter uncertainty instead of just Monte Carlo
    sampling noise.
    """
    base = summarize_distribution(boot_probs)
    stressed = summarize_distribution(boot_probs_stressed)

    p = base["mean"]
    stress_p = stressed["mean"]

    frag = fragility(p, stress_p)

    # Confidence shrinks as the bootstrap spread (sd) grows - a market whose
    # probability swings a lot across re-fits shouldn't score as "robust"
    # even if its central estimate looks attractive.
    calibration_conf = float(np.clip(1.0 - base["sd"] * 6.0, 0.0, 1.0))
    stability = float(np.clip(stress_p, 0.0, 1.0))

    is_trivial = (p >= TRIVIAL_HIGH) or (p <= TRIVIAL_LOW)
    tier = "absolute" if is_trivial else "practical"

    # Only downside risk is penalized - upside (stress *improving* the pick)
    # should not be treated the same as a market that gets worse under stress.
    robust = (
        0.35 * p
        + 0.25 * stress_p
        + 0.15 * calibration_conf
        + 0.15 * stability
        - 0.10 * frag["downside_risk"]
        + 0.05 * frag["upside_potential"]
    )

    return {
        "market": name,
        "line": line,
        "side": side.title(),
        "tier": tier,
        "probability": p,
        "P05": base["P05"],
        "P10": base["P10"],
        "P50": base["P50"],
        "P90": base["P90"],
        "P95": base["P95"],
        "bootstrap_sd": base["sd"],
        "stress_probability": stress_p,
        "downside_risk": frag["downside_risk"],
        "upside_potential": frag["upside_potential"],
        # kept for backward compatibility with anything reading "fragility"
        "fragility": frag["downside_risk"],
        "robust_score": float(robust),
    }


def generate_count_markets(name, lines, boot_by_line, boot_by_line_stressed):
    """boot_by_line / boot_by_line_stressed: dict[line] -> {"over": arr, "under": arr}"""
    rows = []
    for line in lines:
        for side in ("over", "under"):
            rows.append(build_market_row(
                name, line, side,
                boot_by_line[line][side],
                boot_by_line_stressed[line][side],
            ))
    return rows
