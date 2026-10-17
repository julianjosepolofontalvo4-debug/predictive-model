import pandas as pd

from baseline import history_before_match as baseline_history_before_match
from backtest.runner import BacktestRunner


def test_baseline_and_backtest_exclude_same_day_and_future_matches():
    history = pd.DataFrame({
        "fixture_id": [1, 2, 3],
        "date": pd.to_datetime([
            "2026-08-30T20:00:00Z",
            "2026-09-01T20:00:00Z",
            "2026-09-02T20:00:00Z",
        ]),
    })

    target_date = pd.Timestamp("2026-09-01T20:00:00Z")
    available = BacktestRunner.history_before_match(
        history, target_date
    )
    baseline_available = baseline_history_before_match(
        history, target_date
    )

    assert available["fixture_id"].tolist() == [1]
    assert baseline_available["fixture_id"].tolist() == [1]
