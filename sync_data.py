"""CLI for the v0.3 automated data pipeline.

Examples:
  set API_FOOTBALL_KEY=...
  python sync_data.py --league 39 --season 2026 --next 20
  python sync_data.py --league 39 --season 2026 --completed
"""
from __future__ import annotations

import argparse

from football_quant.automation import APIFootballClient, AutoSync, SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=int, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--next", type=int, default=None)
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--completed", action="store_true")
    parser.add_argument("--db", default="data/football_quant.db")
    args = parser.parse_args()

    store = SQLiteStore(args.db)
    sync = AutoSync(APIFootballClient(), store)
    if args.completed:
        count = sync.sync_completed_details(league=args.league, season=args.season)
    else:
        count = sync.sync_fixtures(league=args.league, season=args.season, next=args.next,
                                   from_date=args.from_date, to_date=args.to_date)
    print(f"Sincronizados: {count}")
    print(f"Base de datos: {args.db}")


if __name__ == "__main__":
    main()