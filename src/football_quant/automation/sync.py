from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .api_football import APIFootballClient
from .store import SQLiteStore


@dataclass
class AutoSync:
    client: APIFootballClient
    store: SQLiteStore

    def sync_fixtures(self, *, league: int, season: int, next: int | None = None,
                      from_date: str | None = None, to_date: str | None = None) -> int:
        fixtures = self.client.fixtures(league=league, season=season, next=next,
                                        from_date=from_date, to_date=to_date)
        for fixture in fixtures:
            self.store.upsert_fixture(fixture)
            self.store.save_raw("fixtures", fixture["fixture"]["id"], fixture)
        return len(fixtures)

    def sync_completed_details(self, *, league: int, season: int) -> int:
        fixtures = self.client.fixtures(league=league, season=season, status="FT-AET-PEN")
        count = 0
        for fixture in fixtures:
            fid = int(fixture["fixture"]["id"])
            self.store.upsert_fixture(fixture)
            for kind, fn in (("statistics", self.client.statistics), ("lineups", self.client.lineups), ("events", self.client.events)):
                try:
                    self.store.save_match_payload(fid, kind, fn(fid))
                except RuntimeError:
                    # Coverage varies by competition; preserve the fixture even
                    # when one optional endpoint is unavailable.
                    continue
            count += 1
        return count

    def sync_one(self, fixture_id: int, include_lineups: bool = True) -> dict[str, Any]:
        fixture = self.client.fixture_details(fixture_id)
        if not fixture:
            raise ValueError(f"No existe fixture {fixture_id}")
        self.store.upsert_fixture(fixture)
        result: dict[str, Any] = {"fixture_id": fixture_id}
        result["statistics"] = self.client.statistics(fixture_id)
        self.store.save_match_payload(fixture_id, "statistics", result["statistics"])
        result["events"] = self.client.events(fixture_id)
        self.store.save_match_payload(fixture_id, "events", result["events"])
        if include_lineups:
            try:
                result["lineups"] = self.client.lineups(fixture_id)
                self.store.save_match_payload(fixture_id, "lineups", result["lineups"])
            except RuntimeError:
                result["lineups"] = []
        return result
