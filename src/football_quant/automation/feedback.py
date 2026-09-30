from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from ..engine import FootballQuantEngine
from ..core import temporal_weights
from .store import SQLiteStore

@dataclass
class FeedbackLoop:
    store: SQLiteStore
    engine: FootballQuantEngine

    def drift_report(self,recent_n:int=30,competition:str|None=None)->dict[str,Any]:
        df=self.store.historical_dataframe(competition)
        if len(df)<max(10,recent_n): return {"status":"insufficient_data","matches":len(df)}
        recent=df.tail(recent_n); baseline=df.iloc[:-recent_n] if len(df)>recent_n else df; fields=["home_goals","away_goals","home_corners","away_corners","home_cards","away_cards"]
        drift={}
        for field in fields:
            a,b=baseline[field].mean(),recent[field].mean(); drift[field]={"baseline_mean":float(a),"recent_mean":float(b),"relative_change":float((b-a)/max(abs(a),1e-9))}
        return {"status":"ok","matches":len(df),"recent_n":recent_n,"drift":drift}

    def learning_report(self,competition:str|None=None)->dict[str,Any]:
        return self.store.learning_report(competition)

    def recalibrate_priors(self,*,blend:float=0.20,competition:str|None=None)->dict[str,float]:
        history=self.store.historical_dataframe(competition)
        if len(history)<8: return dict(self.engine.league_prior)
        recent=history.tail(min(80,len(history))); weights=temporal_weights(len(recent)); updated=dict(self.engine.league_prior); blend=max(0.0,min(1.0,float(blend)))
        for key in updated:
            if key in recent:
                recent_mean=float((recent[key].to_numpy(float)*weights).sum())
                updated[key]=(1-blend)*float(updated[key])+blend*recent_mean
        self.engine.league_prior=updated; return updated

    def retrain_snapshot(self,*,league_prior:dict[str,float]|None=None,output_dir:str="data/model_snapshots",competition:str|None=None)->dict[str,Any]:
        history=self.store.historical_dataframe(competition)
        if len(history)<8: return {"status":"insufficient_data","matches":len(history)}
        if league_prior is not None: self.engine.league_prior=dict(league_prior)
        else: self.recalibrate_priors(competition=competition)
        payload=history.to_csv(index=False).encode("utf-8"); version=hashlib.sha256(payload).hexdigest()[:12]; out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); history.to_csv(out/f"history_{version}.csv",index=False)
        metrics={"matches":len(history),"version":version,"competition":competition or "all","features":list(history.columns),"algorithm":"temporal_count_distributions_plus_error_feedback","league_prior":self.engine.league_prior,"learning":self.learning_report(competition)}
        self.store.conn.execute("INSERT OR REPLACE INTO model_versions(version,metrics_json) VALUES (?,?)",(version,json.dumps(metrics))); self.store.conn.commit(); return {"status":"retrained_snapshot",**metrics}
