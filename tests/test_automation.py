import json

from football_quant.automation.store import SQLiteStore


def fixture(status="FT"):
    return {
        "fixture": {"id": 123, "date": "2026-09-01T18:00:00+00:00", "status": {"short": status}, "venue": {"name": "Test Stadium"}},
        "league": {"id": 39, "season": 2026},
        "teams": {"home": {"id": 1, "name": "Home FC"}, "away": {"id": 2, "name": "Away FC"}},
        "goals": {"home": 2, "away": 1},
    }


def stats():
    return [
        {"team": {"name": "Home FC"}, "statistics": [
            {"type": "Corner Kicks", "value": 6}, {"type": "Yellow Cards", "value": 2}, {"type": "Red Cards", "value": 0}
        ]},
        {"team": {"name": "Away FC"}, "statistics": [
            {"type": "Corner Kicks", "value": 4}, {"type": "Yellow Cards", "value": 3}, {"type": "Red Cards", "value": 0}
        ]},
    ]


def test_store_builds_history(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    store.upsert_fixture(fixture())
    store.save_match_payload(123, "statistics", stats())
    df = store.historical_dataframe()
    assert len(df) == 1
    assert df.iloc[0].home_corners == 6
    assert df.iloc[0].away_corners == 4
    assert df.iloc[0].home_cards == 2
    assert df.iloc[0].away_cards == 3
    assert df.iloc[0].away_cards == 3
