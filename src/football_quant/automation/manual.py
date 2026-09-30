from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from football_quant import FootballQuantEngine, MatchInput
from football_quant.automation.store import SQLiteStore, parse_match_date
from football_quant.automation.team_names import normalize_team_name


class ManualMatchManager:
    def __init__(self, store: SQLiteStore, engine: FootballQuantEngine):
        self.store=store; self.engine=engine

    def load_upcoming(self, path: str | Path = "data/upcoming_matches.csv") -> pd.DataFrame:
        df=pd.read_csv(path)
        required=["fixture_id","date","competition","home_team","away_team"]
        missing=[c for c in required if c not in df.columns]
        if missing: raise ValueError(f"Faltan columnas en {path}: {', '.join(missing)}")
        if df["fixture_id"].duplicated().any(): raise ValueError("Hay fixture_id duplicados en el calendario.")
        df["date"]=df["date"].map(lambda x: parse_match_date(x).isoformat())
        df["home_team_input"]=df["home_team"].astype(str).str.strip()
        df["away_team_input"]=df["away_team"].astype(str).str.strip()
        df["home_team"]=df["home_team"].map(normalize_team_name); df["away_team"]=df["away_team"].map(normalize_team_name)
        if (df["home_team"]==df["away_team"]).any(): raise ValueError("Existe un partido con el mismo equipo local y visitante.")
        return df

    def predict_upcoming(self, path: str | Path = "data/upcoming_matches.csv", *, n_simulations:int=20000, n_bootstrap:int=200, seed:int=42) -> list[dict[str,Any]]:
        upcoming=self.load_upcoming(path); outputs=[]
        for row in upcoming.itertuples(index=False):
            match_date=parse_match_date(row.date); competition=str(row.competition).strip()
            history=self.store.historical_dataframe(competition)
            if len(history)<8:
                raise ValueError(f"{competition}: solo hay {len(history)} partidos completos antes de predecir {row.home_team} vs {row.away_team}. Importa más historial de esta competición.")
            history_dates=pd.to_datetime(history["date"],utc=True,format="mixed")
            match_history=history[history_dates<match_date].copy()
            if len(match_history)<8:
                raise ValueError(f"{competition}: solo hay {len(match_history)} partidos anteriores a {row.home_team} vs {row.away_team}; se necesitan al menos 8.")
            home_team=normalize_team_name(row.home_team); away_team=normalize_team_name(row.away_team)
            bias,learning=self.store.learning_bias(competition,before_date=match_date)
            base_prior=dict(self.engine.league_prior)
            learned_prior={k:max(0.05,float(base_prior[k])+float(bias.get(k,0.0))) for k in base_prior}
            local_engine=FootballQuantEngine(learned_prior,decay=self.engine.decay,shrink_k=self.engine.shrink_k)
            result=local_engine.fit_predict(match_history,MatchInput(home_team,away_team),n_simulations=n_simulations,n_bootstrap=n_bootstrap,seed=seed)
            prediction=self._serialize_result(result)
            latest_date=pd.to_datetime(match_history["date"],utc=True,format="mixed").max()
            days_since=max(0.0,float((match_date-latest_date).total_seconds()/86400.0)); freshness=max(0.0,min(1.0,2.718281828**(-days_since/45.0)))
            prediction["metadata"].update({"model_version":"manual-v0.4-learning","competition":competition,"learning":learning,"learned_bias":bias,"effective_prior":learned_prior,"freshness_score":freshness,"data_quality":float(min(prediction["metadata"].get("data_quality",1.0),freshness))})
            prediction.update({"home_team":home_team,"away_team":away_team,"fixture_id":int(row.fixture_id),"date":match_date.isoformat(),"competition":competition,"history_matches_used":int(len(match_history)),"latest_history_date":latest_date.isoformat(),"days_since_latest_history":round(days_since,2),"team_form":{home_team:self.store.team_form(home_team,competition=competition),away_team:self.store.team_form(away_team,competition=competition)},"generated_at":datetime.now(timezone.utc).isoformat()})
            # Backward-compatible display keys for aliases such as Villarreal/Real Betis.
            if str(getattr(row, "home_team_input", home_team)) != home_team:
                prediction["team_form"][str(row.home_team_input)] = prediction["team_form"][home_team]
            if str(getattr(row, "away_team_input", away_team)) != away_team:
                prediction["team_form"][str(row.away_team_input)] = prediction["team_form"][away_team]
            self.store.upsert_manual_fixture(int(row.fixture_id),match_date.isoformat(),competition,home_team,away_team); self.store.save_prediction(int(row.fixture_id),prediction); outputs.append(prediction)
        return outputs

    def record_result(self, fixture_id:int, observed:dict[str,Any])->dict[str,Any]:
        prediction=self.store.get_prediction(fixture_id)
        if prediction is None: raise ValueError(f"No existe una predicción guardada para fixture_id {fixture_id}.")
        self.store.complete_manual_fixture(fixture_id,observed)
        return self.store.update_prediction_observation(fixture_id,observed)

    @staticmethod
    def _serialize_result(result)->dict[str,Any]:
        return {"expected_values":{k:float(v) for k,v in result.expected_values.items()},"parameters":result.parameters,"metadata":result.metadata,"markets":result.market_table.to_dict(orient="records")}
