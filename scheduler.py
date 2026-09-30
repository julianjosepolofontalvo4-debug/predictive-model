"""Simple autonomous polling loop.

Use an OS scheduler (Task Scheduler/cron) for production. This loop is useful
for local development and keeps the API interval explicit.
"""
from __future__ import annotations

import argparse
import time

from football_quant import FootballQuantEngine
from football_quant.automation import APIFootballClient, AutoSync, FeedbackLoop, SQLiteStore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--league", type=int, required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--interval", type=int, default=900)
    p.add_argument("--db", default="data/football_quant.db")
    args = p.parse_args()

    store = SQLiteStore(args.db)
    sync = AutoSync(APIFootballClient(), store)
    engine = FootballQuantEngine({"home_goals": 1.45, "away_goals": 1.15, "home_corners": 5.2, "away_corners": 4.4, "home_cards": 2.1, "away_cards": 2.2})
    feedback = FeedbackLoop(store, engine)
    while True:
        try:
            print("Sync upcoming fixtures...")
            sync.sync_fixtures(league=args.league, season=args.season, next=20)
            print("Sync completed matches...")
            sync.sync_completed_details(league=args.league, season=args.season)
            print("Recalibrating model snapshot...")
            print(feedback.retrain_snapshot())
        except Exception as exc:
            print(f"SYNC ERROR: {exc}")
        time.sleep(max(args.interval, 60))


if __name__ == "__main__":
    main()
