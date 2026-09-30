import pandas as pd
from football_quant import FootballQuantEngine, MatchInput

history = pd.read_csv("data/sample_matches.csv")
engine = FootballQuantEngine({
    "home_goals": 1.45, "away_goals": 1.15,
    "home_corners": 5.2, "away_corners": 4.4,
    "home_cards": 2.1, "away_cards": 2.2,
})

result = engine.fit_predict(
    history, MatchInput("Home FC", "Away FC"),
    n_simulations=20000, n_bootstrap=200, seed=42
)

print("EXPECTED VALUES")
for key, value in result.expected_values.items():
    print(f"{key:20s}: {value:.3f}")

cols = [
    "market", "line", "side", "probability",
    "P05", "P10", "P50", "P95",
    "stress_probability", "downside_risk", "upside_potential", "robust_score"
]

print("\nABSOLUTE (near-certain lines - sanity check only, not decision-grade)")
print(result.absolute_markets(8)[cols].to_string(index=False))

print("\nPRACTICAL (non-trivial lines, ranked by raw probability)")
print(result.practical_markets(10)[cols].to_string(index=False))

print("\nROBUST (non-trivial lines, ranked by robust_score = stability under stress)")
print(result.robust_markets(10)[cols].to_string(index=False))

print("\nMETADATA")
for key, value in result.metadata.items():
    print(f"{key}: {value}")
