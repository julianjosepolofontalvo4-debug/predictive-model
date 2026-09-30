from __future__ import annotations

import argparse

from football_quant import FootballQuantEngine
from football_quant.automation import SQLiteStore, FeedbackLoop
from football_quant.automation.manual import ManualMatchManager


PRIOR = {
    "home_goals": 1.45, "away_goals": 1.15,
    "home_corners": 5.2, "away_corners": 4.4,
    "home_cards": 2.1, "away_cards": 2.2,
}

FIELDS = [
    ("home_goals", "Goles local"), ("away_goals", "Goles visitante"),
    ("home_corners", "Corners local"), ("away_corners", "Corners visitante"),
    ("home_cards", "Tarjetas local"), ("away_cards", "Tarjetas visitante"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Registra el resultado real de un partido ya predicho.")
    parser.add_argument("fixture_id", type=int, help="ID del partido en data/upcoming_matches.csv")
    parser.add_argument("--db", default="data/football_quant.db")
    args = parser.parse_args()

    values = {}
    for key, label in FIELDS:
        raw = input(f"{label}: ").strip()
        try:
            values[key] = float(raw)
        except ValueError as exc:
            raise SystemExit(f"Valor inválido para {label}: {raw}") from exc

    store = SQLiteStore(args.db)
    try:
        manager = ManualMatchManager(store, FootballQuantEngine(PRIOR))
        audit = manager.record_result(args.fixture_id, values)
        feedback = FeedbackLoop(store, FootballQuantEngine(PRIOR))
        recalibration = feedback.recalibrate_priors()
        snapshot = feedback.retrain_snapshot()
        print("\n✓ Resultado almacenado")
        print("✓ Predicción auditada")
        print("✓ Estadísticas incorporadas al historial")
        print("\nERRORES (real - esperado)")
        for key, value in audit["error"].items():
            print(f"  {key:18s}: {value:+.3f}")
        print("\n✓ Feedback loop ejecutado")
        print(f"✓ Recalibración: {recalibration}")
        print(f"✓ Snapshot: {snapshot.get("status")}")
        print("\nEl próximo `python predict.py` ya podrá utilizar este partido como información histórica.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
