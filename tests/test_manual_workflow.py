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


def record_barcelona_result(store, fixture_id, date, home_goals, away_goals):
    store.upsert_manual_fixture(
        fixture_id, date, "LaLiga", "FC Barcelona", "Test Opponent"
    )
    store.complete_manual_fixture(fixture_id, {
        "home_goals": home_goals, "away_goals": away_goals,
        "home_corners": 5, "away_corners": 4,
        "home_cards": 1, "away_cards": 2,
    })


def test_barcelona_form_fills_current_season_from_previous_and_excludes_future(tmp_path):
    store = SQLiteStore(tmp_path / "barcelona.db")
    previous_season = [
        ("2026-05-01T20:00:00Z", 0, 1),
        ("2026-05-09T20:00:00Z", 2, 0),
        ("2026-05-17T20:00:00Z", 1, 1),
        ("2026-05-25T20:00:00Z", 0, 1),
    ]
    current_season = [
        ("2026-08-10T20:00:00Z", 3, 0),
        ("2026-08-20T20:00:00Z", 2, 2),
        ("2026-08-28T20:00:00Z", 1, 0),
    ]
    for fixture_id, (date, home_goals, away_goals) in enumerate(
        previous_season + current_season, start=920000
    ):
        record_barcelona_result(store, fixture_id, date, home_goals, away_goals)
    record_barcelona_result(store, 920007, "2026-09-01T21:00:00Z", 0, 1)
    record_barcelona_result(store, 920008, "2026-09-03T20:00:00Z", 0, 1)

    form = store.team_form(
        "Barcelona", before_date="2026-09-01T20:00:00Z"
    )

    assert form["matches"] == 5
    assert form["last_results"] == ["D", "L", "W", "D", "W"]
    assert form["points"] == 8


def test_barcelona_form_uses_only_five_current_season_matches(tmp_path):
    store = SQLiteStore(tmp_path / "barcelona_five.db")
    for fixture_id, date in enumerate(
        ["2026-05-01T20:00:00Z", "2026-05-09T20:00:00Z",
         "2026-05-17T20:00:00Z", "2026-05-25T20:00:00Z"],
        start=930000,
    ):
        record_barcelona_result(store, fixture_id, date, 0, 1)
    for fixture_id, date in enumerate(
        ["2026-08-10T20:00:00Z", "2026-08-15T20:00:00Z",
         "2026-08-20T20:00:00Z", "2026-08-25T20:00:00Z",
         "2026-08-30T20:00:00Z"],
        start=930004,
    ):
        record_barcelona_result(store, fixture_id, date, 2, 0)

    form = store.team_form(
        "FC Barcelona", n=10, before_date="2026-09-01T20:00:00Z"
    )

    assert form["matches"] == 5
    assert form["points"] == 15
    assert form["last_results"] == ["W", "W", "W", "W", "W"]


def test_prediction_history_excludes_same_day_after_cutoff_and_future(tmp_path):
    store = make_store(tmp_path)
    original_history_count = len(store.historical_dataframe("Test League"))
    for fixture_id, date in [
        (940001, "2026-12-01T21:00:00Z"),
        (940002, "2026-12-02T20:00:00Z"),
    ]:
        store.upsert_manual_fixture(
            fixture_id, date, "Test League", "Sevilla FC", "Rayo Vallecano"
        )
        store.complete_manual_fixture(fixture_id, {
            "home_goals": 3, "away_goals": 0,
            "home_corners": 7, "away_corners": 2,
            "home_cards": 1, "away_cards": 3,
        })

    upcoming = tmp_path / "cutoff.csv"
    upcoming.write_text(
        "fixture_id,date,competition,home_team,away_team\n"
        "940003,2026-12-01T20:00:00Z,Test League,Sevilla FC,Rayo Vallecano\n",
        encoding="utf-8",
    )
    prediction = ManualMatchManager(store, make_engine()).predict_upcoming(
        upcoming, n_simulations=100, n_bootstrap=5
    )[0]

    assert prediction["history_matches_used"] == original_history_count
    assert prediction["latest_history_date"] == "2026-09-13T00:00:00+00:00"
