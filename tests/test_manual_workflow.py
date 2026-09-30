import pandas as pd

from football_quant import FootballQuantEngine
from football_quant.automation import SQLiteStore
from football_quant.automation.manual import ManualMatchManager


def make_store(tmp_path):
    store = SQLiteStore(tmp_path / "workflow.db")
    history = pd.read_csv("data/sample_matches.csv")
    for i, row in history.iterrows():
        fixture_id = 500000 + i
        store.upsert_manual_fixture(
            fixture_id, str(pd.Timestamp(row.date)), "Test League", row.home_team, row.away_team
        )
        store.complete_manual_fixture(fixture_id, {
            "home_goals": row.home_goals, "away_goals": row.away_goals,
            "home_corners": row.home_corners, "away_corners": row.away_corners,
            "home_cards": row.home_cards, "away_cards": row.away_cards,
        })
    return store


def make_engine():
    return FootballQuantEngine({
        "home_goals": 1.45, "away_goals": 1.15,
        "home_corners": 5.2, "away_corners": 4.4,
        "home_cards": 2.1, "away_cards": 2.2,
    })


def test_prediction_then_result_feedback(tmp_path):
    store = make_store(tmp_path)
    manager = ManualMatchManager(store, make_engine())
    upcoming = tmp_path / "upcoming.csv"
    upcoming.write_text(
        "fixture_id,date,competition,home_team,away_team\n"
        "900001,2026-12-01T20:00:00+00:00,Test League,Sevilla FC,Rayo Vallecano\n",
        encoding="utf-8",
    )

    predictions = manager.predict_upcoming(upcoming, n_simulations=500, n_bootstrap=10)
    assert len(predictions) == 1
    assert store.conn.execute("SELECT COUNT(*) FROM prediction_audits").fetchone()[0] == 1

    audit = manager.record_result(900001, {
        "home_goals": 2, "away_goals": 0,
        "home_corners": 5, "away_corners": 3,
        "home_cards": 2, "away_cards": 1,
    })
    assert "home_goals" in audit["error"]
    assert len(store.historical_dataframe()) == len(pd.read_csv("data/sample_matches.csv")) + 1
    row = store.conn.execute(
        "SELECT observed_json, error_json FROM prediction_audits WHERE fixture_id=900001"
    ).fetchone()
    assert row["observed_json"] is not None
    assert row["error_json"] is not None


def test_team_name_aliases_are_normalized(tmp_path):
    store = make_store(tmp_path)
    manager = ManualMatchManager(store, make_engine())
    upcoming = tmp_path / "upcoming_alias.csv"
    upcoming.write_text(
        "fixture_id,date,competition,home_team,away_team\n"
        "900002,2026-12-01T20:00:00+00:00,Test League,Villarreal,Real Betis\n",
        encoding="utf-8",
    )

    predictions = manager.predict_upcoming(upcoming, n_simulations=100, n_bootstrap=5)
    assert len(predictions) == 1
    assert predictions[0]["team_form"]["Villarreal"]["matches"] > 0
    assert predictions[0]["team_form"]["Real Betis"]["matches"] > 0


def test_competition_isolation_and_mixed_dates(tmp_path):
    store = SQLiteStore(tmp_path / "iso.db")
    store.upsert_manual_fixture(1, "2026-01-01T20:00:00-05:00", "Premier", "Man Utd", "Man City")
    store.complete_manual_fixture(1, {"home_goals":1,"away_goals":2,"home_corners":4,"away_corners":5,"home_cards":2,"away_cards":3})
    df = store.historical_dataframe("Premier")
    assert df.iloc[0].home_team == "Manchester United"
    assert df.iloc[0].away_team == "Manchester City"
    assert str(df.iloc[0].date).endswith("+00:00")


def test_error_learning_is_conservative(tmp_path):
    store = make_store(tmp_path)
    manager = ManualMatchManager(store, make_engine())
    upcoming = tmp_path / "u.csv"
    upcoming.write_text("fixture_id,date,competition,home_team,away_team\n910000,2027-01-01T20:00:00-05:00,Test League,Sevilla FC,Rayo Vallecano\n", encoding="utf-8")
    for i in range(5):
        fid=910000+i
        up=tmp_path/f"u{i}.csv"
        up.write_text(f"fixture_id,date,competition,home_team,away_team\n{fid},2027-01-01T20:00:00-05:00,Test League,Sevilla FC,Rayo Vallecano\n",encoding="utf-8")
        manager.predict_upcoming(up,n_simulations=100,n_bootstrap=5)
        manager.record_result(fid,{"home_goals":4,"away_goals":0,"home_corners":8,"away_corners":1,"home_cards":4,"away_cards":0})
    bias, meta = store.learning_bias("Test League", before_date="2028-01-01T00:00:00Z")
    assert meta["status"] == "ok"
    assert abs(bias["home_goals"]) <= 0.75

