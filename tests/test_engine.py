import pandas as pd
from football_quant.engine import FootballQuantEngine, MatchInput

def make_engine():
    return FootballQuantEngine({
        "home_goals": 1.45,
        "away_goals": 1.15,
        "home_corners": 5.2,
        "away_corners": 4.4,
        "home_cards": 2.1,
        "away_cards": 2.2,
    })

def test_engine_runs():
    history = pd.read_csv("data/sample_matches.csv")
    result = make_engine().fit_predict(
        history, MatchInput("Sevilla FC", "Rayo Vallecano"),
        n_simulations=2000, n_bootstrap=50, seed=42
    )
    assert not result.market_table.empty
    assert result.market_table["probability"].between(0, 1).all()
    assert result.market_table["stress_probability"].between(0, 1).all()
    assert result.expected_values["home_goals"] >= 0

def test_reproducibility():
    history = pd.read_csv("data/sample_matches.csv")
    engine = make_engine()
    r1 = engine.fit_predict(history, MatchInput("Sevilla FC", "Rayo Vallecano"), 1000, 30, 7)
    r2 = engine.fit_predict(history, MatchInput("Sevilla FC", "Rayo Vallecano"), 1000, 30, 7)
    assert r1.market_table["probability"].tolist() == r2.market_table["probability"].tolist()
