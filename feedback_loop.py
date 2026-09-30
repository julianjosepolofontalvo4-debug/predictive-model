"""Run the local feedback/recalibration loop after data synchronization."""
from __future__ import annotations

import argparse

from football_quant import FootballQuantEngine
from football_quant.automation import FeedbackLoop, SQLiteStore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/football_quant.db")
    args = p.parse_args()
    prior = {
        "home_goals": 1.45, "away_goals": 1.15,
        "home_corners": 5.2, "away_corners": 4.4,
        "home_cards": 2.1, "away_cards": 2.2,
    }
    store = SQLiteStore(args.db)
    loop = FeedbackLoop(store, FootballQuantEngine(prior))
    print("DRIFT:", loop.drift_report())
    print("PREDICTION ERRORS:", loop.learning_report())
    print("RETRAIN:", loop.retrain_snapshot())


if __name__ == "__main__":
    main()

