from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class APIFootballClient:
    api_key: str | None = None
    base_url: str = "https://v3.football.api-sports.io"
    timeout: int = 30
    retries: int = 3

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("API_FOOTBALL_KEY") or os.getenv("APISPORTS_KEY")
        if not self.api_key:
            raise ValueError("Configura API_FOOTBALL_KEY en el entorno antes de usar APIFootballClient.")
        self.session = requests.Session()
        self.session.headers.update({"x-apisports-key": self.api_key, "Accept": "application/json"})

    def get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, params={k: v for k, v in params.items() if v is not None}, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if payload.get("errors"):
                    raise RuntimeError(f"API-Football errors: {payload['errors']}")
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"API-Football request failed: {url}") from last_error

    def fixtures(self, *, league: int, season: int, next: int | None = None,
                 from_date: str | None = None, to_date: str | None = None,
                 status: str | None = None) -> list[dict[str, Any]]:
        payload = self.get("fixtures", league=league, season=season, next=next,
                           **{"from": from_date, "to": to_date}, status=status)
        return payload.get("response", [])

    def fixture_details(self, fixture_id: int) -> dict[str, Any] | None:
        rows = self.get("fixtures", id=fixture_id).get("response", [])
        return rows[0] if rows else None

    def statistics(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.get("fixtures/statistics", fixture=fixture_id).get("response", [])

    def lineups(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.get("fixtures/lineups", fixture=fixture_id).get("response", [])

    def events(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.get("fixtures/events", fixture=fixture_id).get("response", [])

    def injuries(self, *, league: int, season: int, team: int | None = None) -> list[dict[str, Any]]:
        return self.get("injuries", league=league, season=season, team=team).get("response", [])
