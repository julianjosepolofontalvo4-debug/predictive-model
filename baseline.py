from __future__ import annotations

import pandas as pd
import numpy as np


class BaselineModel:

    def __init__(self):
        self.means = {}

    def fit(self, history: pd.DataFrame):

        self.means = {
            "home_goals": history["home_goals"].mean(),
            "away_goals": history["away_goals"].mean(),

            "home_corners": history["home_corners"].mean(),
            "away_corners": history["away_corners"].mean(),

            "home_cards": history["home_cards"].mean(),
            "away_cards": history["away_cards"].mean(),
        }

        return self

    def predict(self):

        home_goals = self.means["home_goals"]
        away_goals = self.means["away_goals"]

        home_corners = self.means["home_corners"]
        away_corners = self.means["away_corners"]

        home_cards = self.means["home_cards"]
        away_cards = self.means["away_cards"]

        return {
            "pred_home_goals": home_goals,
            "pred_away_goals": away_goals,
            "pred_total_goals": home_goals + away_goals,

            "pred_home_corners": home_corners,
            "pred_away_corners": away_corners,
            "pred_total_corners": home_corners + away_corners,

            "pred_home_cards": home_cards,
            "pred_away_cards": away_cards,
            "pred_total_cards": home_cards + away_cards,
        }


def mae(actual, predicted):
    return float(
        np.mean(
            np.abs(
                np.asarray(actual)
                -
                np.asarray(predicted)
            )
        )
    )


def rmse(actual, predicted):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(actual)
                    -
                    np.asarray(predicted)
                ) ** 2
            )
        )
    )


def run_baseline(csv_path: str, min_train_matches: int = 8):

    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    results = []

    for i in range(len(df)):

        history = df.iloc[:i].copy()
        match = df.iloc[i]

        if len(history) < min_train_matches:
            continue

        model = BaselineModel()

        model.fit(history)

        prediction = model.predict()

        results.append({

            "date": match["date"],

            "home_team": match["home_team"],

            "away_team": match["away_team"],

            "pred_total_goals":
                prediction["pred_total_goals"],

            "actual_total_goals":
                (
                    match["home_goals"]
                    +
                    match["away_goals"]
                ),

            "pred_total_corners":
                prediction["pred_total_corners"],

            "actual_total_corners":
                (
                    match["home_corners"]
                    +
                    match["away_corners"]
                ),

            "pred_total_cards":
                prediction["pred_total_cards"],

            "actual_total_cards":
                (
                    match["home_cards"]
                    +
                    match["away_cards"]
                ),
        })

    results = pd.DataFrame(results)

    if results.empty:
        print("No hay suficientes datos.")
        return

    # ==========================================
    # ERRORES
    # ==========================================

    goal_mae = mae(
        results["actual_total_goals"],
        results["pred_total_goals"]
    )

    corner_mae = mae(
        results["actual_total_corners"],
        results["pred_total_corners"]
    )

    card_mae = mae(
        results["actual_total_cards"],
        results["pred_total_cards"]
    )

    goal_rmse = rmse(
        results["actual_total_goals"],
        results["pred_total_goals"]
    )

    corner_rmse = rmse(
        results["actual_total_corners"],
        results["pred_total_corners"]
    )

    card_rmse = rmse(
        results["actual_total_cards"],
        results["pred_total_cards"]
    )

    print()
    print("=" * 70)
    print("BASELINE MODEL")
    print("=" * 70)

    print(
        f"Partidos evaluados: {len(results)}"
    )

    print()
    print("GOLES")

    print(
        f"MAE:  {goal_mae:.4f}"
    )

    print(
        f"RMSE: {goal_rmse:.4f}"
    )

    print()
    print("CORNERS")

    print(
        f"MAE:  {corner_mae:.4f}"
    )

    print(
        f"RMSE: {corner_rmse:.4f}"
    )

    print()
    print("TARJETAS")

    print(
        f"MAE:  {card_mae:.4f}"
    )

    print(
        f"RMSE: {card_rmse:.4f}"
    )

    results.to_csv(
        "data/baseline_results.csv",
        index=False
    )

    print()
    print(
        "Resultados guardados en:"
    )

    print(
        "data/baseline_results.csv"
    )


if __name__ == "__main__":

    run_baseline(
        "data/sample_matches.csv",
        min_train_matches=8
    )