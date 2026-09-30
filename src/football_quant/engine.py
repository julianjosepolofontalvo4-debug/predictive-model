from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .core import fit_count_distribution, temporal_weights
from .markets import generate_count_markets
from .simulation import quantiles, simulate_match
from .uncertainty import data_quality_score

STAT_COLUMNS = ("home_goals", "away_goals", "home_corners", "away_corners", "home_cards", "away_cards")

DERIVED_DEFINITIONS = {
    "Goals Total": ("home_goals", "away_goals"),
    "Corners Total": ("home_corners", "away_corners"),
    "Cards Total": ("home_cards", "away_cards"),
}

LINE_GRID = {
    "Goals Total": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    "Corners Total": [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
    "Cards Total": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
}


@dataclass
class MatchInput:
    home_team: str
    away_team: str


@dataclass
class EngineResult:
    parameters: dict
    expected_values: dict
    market_table: pd.DataFrame
    simulations: dict
    metadata: dict

    def absolute_markets(self, top: int = 10) -> pd.DataFrame:
        """Near-certain lines (very high/low probability). Useful as a sanity
        check, not as a betting/decision list - see the docstring in markets.py."""
        df = self.market_table[self.market_table["tier"] == "absolute"]
        return df.sort_values("probability", ascending=False).head(top)

    def practical_markets(self, top: int = 10) -> pd.DataFrame:
        """Non-trivial lines (probability strictly between the trivial
        thresholds), ranked by raw probability."""
        df = self.market_table[self.market_table["tier"] == "practical"]
        return df.sort_values("probability", ascending=False).head(top)

    def robust_markets(self, top: int = 10) -> pd.DataFrame:
        """Non-trivial lines ranked by robust_score: probability weighted
        against stability under stress and bootstrap confidence."""
        df = self.market_table[self.market_table["tier"] == "practical"]
        return df.sort_values("robust_score", ascending=False).head(top)


class FootballQuantEngine:
    REQUIRED = {
        "date", "home_team", "away_team",
        "home_goals", "away_goals",
        "home_corners", "away_corners",
        "home_cards", "away_cards",
    }

    def __init__(self, league_prior: dict[str, float], decay: float = 0.08, shrink_k: float = 5.0):
        self.league_prior = league_prior
        self.decay = decay
        self.shrink_k = shrink_k

    def _validate(self, df):
        missing = sorted(self.REQUIRED - set(df.columns))
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {missing}")
        if len(df) < 8:
            raise ValueError("Se necesitan al menos 8 partidos históricos para V1.")

    def _team_subset(self, df, team):
        matches = df[(df.home_team == team) | (df.away_team == team)]
        sort_columns = ["date"]
        if "fixture_id" in matches.columns:
            sort_columns.append("fixture_id")
        return matches.sort_values(sort_columns, kind="mergesort")

    def _team_splits(self, history, match):
        """Return the (home-venue history for home team, away-venue history
        for away team) frames plus their temporal weights, with the same
        backoff logic used everywhere else in the engine."""
        home_hist = self._team_subset(history, match.home_team)
        away_hist = self._team_subset(history, match.away_team)

        hh = home_hist[home_hist.home_team == match.home_team]
        aa = away_hist[away_hist.away_team == match.away_team]

        if len(hh) < 4:
            hh = home_hist
        if len(aa) < 4:
            aa = away_hist

        wh = temporal_weights(len(hh), self.decay)
        wa = temporal_weights(len(aa), self.decay)
        return hh, wh, aa, wa

    def _fit_from_frames(self, hh, wh, aa, wa):
        mappings = {
            "home_goals": (hh["home_goals"].to_numpy(float), wh),
            "away_goals": (aa["away_goals"].to_numpy(float), wa),
            "home_corners": (hh["home_corners"].to_numpy(float), wh),
            "away_corners": (aa["away_corners"].to_numpy(float), wa),
            "home_cards": (hh["home_cards"].to_numpy(float), wh),
            "away_cards": (aa["away_cards"].to_numpy(float), wa),
        }
        return {
            key: fit_count_distribution(values, weights, self.league_prior[key], self.shrink_k)
            for key, (values, weights) in mappings.items()
        }

    def _rates(self, history, match):
        hh, wh, aa, wa = self._team_splits(history, match)
        return self._fit_from_frames(hh, wh, aa, wa)

    def _bootstrap_market_distributions(self, history, match, n_bootstrap, n_mini_sim, stress_pct, rng):
        """The core statistical fix: instead of resampling the 100k Monte
        Carlo draws (which only captures simulation noise around a single
        fixed set of parameters), resample the *historical matches* with
        replacement, refit the count distributions on each resample, and
        re-run a small Monte Carlo simulation from the refit parameters.
        The spread across replicates then reflects "how much would this
        prediction change if the season had gone slightly differently" -
        genuine model/parameter uncertainty - rather than just simulation
        noise around one fixed estimate.
        """
        hh, wh, aa, wa = self._team_splits(history, match)
        n_h, n_a = len(hh), len(aa)

        hh_np = {col: hh[col].to_numpy(float) for col in ("home_goals", "home_corners", "home_cards")}
        aa_np = {col: aa[col].to_numpy(float) for col in ("away_goals", "away_corners", "away_cards")}
        effective_n_h = 1.0 / np.sum(wh ** 2)
        effective_n_a = 1.0 / np.sum(wa ** 2)

        boot = {name: {line: {"over": [], "under": []} for line in lines} for name, lines in LINE_GRID.items()}
        boot_stressed = {name: {line: {"over": [], "under": []} for line in lines} for name, lines in LINE_GRID.items()}

        for _ in range(n_bootstrap):
            idx_h = rng.choice(n_h, size=n_h, replace=True, p=wh)
            idx_a = rng.choice(n_a, size=n_a, replace=True, p=wa)

            # Sampling already reflects recency. Equal weights within each
            # replicate avoid applying temporal weights twice.
            wh_b = np.full(n_h, 1.0 / n_h)
            wa_b = np.full(n_a, 1.0 / n_a)

            mappings = {
                "home_goals": (hh_np["home_goals"][idx_h], wh_b),
                "away_goals": (aa_np["away_goals"][idx_a], wa_b),
                "home_corners": (hh_np["home_corners"][idx_h], wh_b),
                "away_corners": (aa_np["away_corners"][idx_a], wa_b),
                "home_cards": (hh_np["home_cards"][idx_h], wh_b),
                "away_cards": (aa_np["away_cards"][idx_a], wa_b),
            }
            params_b = {
                key: fit_count_distribution(
                    values,
                    weights,
                    self.league_prior[key],
                    self.shrink_k,
                    effective_n_override=(
                        effective_n_h if key.startswith("home_") else effective_n_a
                    ),
                )
                for key, (values, weights) in mappings.items()
            }
            sim_params_b = {
                k: {"mean": v["mean"], "dispersion": v["dispersion_index"], "model": v["model"]}
                for k, v in params_b.items()
            }

            seed_b = int(rng.integers(0, 2**31 - 1))
            sims_b = simulate_match(sim_params_b, n=n_mini_sim, seed=seed_b)
            sims_b_stressed = simulate_match(
                sim_params_b, n=n_mini_sim, seed=seed_b + 1,
                shocks={k: 1.0 - stress_pct for k in sim_params_b},
            )

            for name, (a_key, b_key) in DERIVED_DEFINITIONS.items():
                vals = sims_b[a_key] + sims_b[b_key]
                vals_s = sims_b_stressed[a_key] + sims_b_stressed[b_key]
                for line in LINE_GRID[name]:
                    boot[name][line]["over"].append(float(np.mean(vals > line)))
                    boot[name][line]["under"].append(float(np.mean(vals < line)))
                    boot_stressed[name][line]["over"].append(float(np.mean(vals_s > line)))
                    boot_stressed[name][line]["under"].append(float(np.mean(vals_s < line)))

        return boot, boot_stressed

    def fit_predict(self, history, match, n_simulations=100_000, n_bootstrap=500, seed=42, stress_pct=0.10):
        history = history.copy()
        history["date"] = pd.to_datetime(history["date"], utc=True, format="mixed")
        sort_columns = ["date"]
        if "fixture_id" in history.columns:
            sort_columns.append("fixture_id")
        history = history.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
        self._validate(history)

        params = self._rates(history, match)
        sim_params = {
            k: {"mean": v["mean"], "dispersion": v["dispersion_index"], "model": v["model"]}
            for k, v in params.items()
        }

        # This large single-parameter simulation is only used for the
        # point "expected value" summary and the P05-P95 of the raw counts
        # (goals/corners/cards) - it is NOT used anymore to size the
        # market-probability uncertainty bands, since that conflated
        # simulation noise with real model uncertainty.
        base = simulate_match(sim_params, n=n_simulations, seed=seed)
        expected = {k: float(np.mean(v)) for k, v in base.items()}
        derived = {
            name: base[a] + base[b] for name, (a, b) in DERIVED_DEFINITIONS.items()
        }

        rng = np.random.default_rng(seed)
        n_mini_sim = max(500, n_simulations // 50)
        boot, boot_stressed = self._bootstrap_market_distributions(
            history, match, n_bootstrap=n_bootstrap, n_mini_sim=n_mini_sim,
            stress_pct=stress_pct, rng=rng,
        )

        rows = []
        for name, lines in LINE_GRID.items():
            rows.extend(generate_count_markets(name, lines, boot[name], boot_stressed[name]))

        table = pd.DataFrame(rows).sort_values(
            ["robust_score", "probability"], ascending=False
        ).reset_index(drop=True)

        quality = data_quality_score(
            n_matches=len(history),
            freshness_score=1.0,
            split_score=0.8,
            source_score=1.0,
        )

        metadata = {
            "home_team": match.home_team,
            "away_team": match.away_team,
            "historical_matches": len(history),
            "simulation_count": n_simulations,
            "bootstrap_count": n_bootstrap,
            "bootstrap_mini_sim": n_mini_sim,
            "bootstrap_method": "dataset_resample_refit",
            "stress_pct": stress_pct,
            "data_quality": quality,
            "uncertainty": "moderate" if quality >= 0.70 else "high",
            "distribution_quantiles": {k: quantiles(v) for k, v in derived.items()},
            "known_limitations": [
                "no_dixon_coles_low_score_correction",
                "no_dynamic_opponent_strength_rating",
                "no_historical_calibration_check",
            ],
        }

        return EngineResult(params, expected, table, base, metadata)
