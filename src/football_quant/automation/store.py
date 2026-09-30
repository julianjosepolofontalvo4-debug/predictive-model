from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .team_names import normalize_team_name


STAT_FIELDS = ("home_goals", "away_goals", "home_corners", "away_corners", "home_cards", "away_cards")


def parse_match_date(value: Any) -> pd.Timestamp:
    """Strict, timezone-safe parser for all supported match-date formats."""
    try:
        ts = pd.to_datetime(value, utc=True, format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Fecha inválida: {value!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"Fecha inválida: {value!r}")
    return ts


class SQLiteStore:
    """Persistent store for fixtures, predictions, observations and learning."""

    def __init__(self, path: str | Path = "data/football_quant.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id INTEGER PRIMARY KEY, league_id INTEGER, season INTEGER,
                date TEXT NOT NULL, status TEXT, home_id INTEGER, home_team TEXT NOT NULL,
                away_id INTEGER, away_team TEXT NOT NULL, home_goals REAL, away_goals REAL,
                venue TEXT, competition TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS match_stats (fixture_id INTEGER PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS lineups (fixture_id INTEGER PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS events (fixture_id INTEGER PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS raw_api (id INTEGER PRIMARY KEY AUTOINCREMENT, endpoint TEXT NOT NULL, object_id TEXT, payload_json TEXT NOT NULL, fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS prediction_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT, fixture_id INTEGER, prediction_json TEXT NOT NULL,
                observed_json TEXT, error_json TEXT, model_version TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, observed_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_audits_fixture ON prediction_audits(fixture_id);
            CREATE TABLE IF NOT EXISTS model_versions (version TEXT PRIMARY KEY, metrics_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            """
        )
        for table, column, definition in [
            ("fixtures", "competition", "TEXT"),
            ("prediction_audits", "error_json", "TEXT"),
            ("prediction_audits", "model_version", "TEXT"),
            ("prediction_audits", "observed_at", "TEXT"),
        ]:
            self._ensure_column(table, column, definition)
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cols = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _stat_value(stats: dict[str, Any], name: str) -> float | None:
        value = stats.get(name)
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def upsert_fixture(self, fixture: dict[str, Any]) -> None:
        fid = int(fixture["fixture"]["id"]); f = fixture["fixture"]; teams = fixture["teams"]
        goals = fixture.get("goals") or {}; league = fixture.get("league") or {}
        self.conn.execute(
            """INSERT INTO fixtures(fixture_id,league_id,season,date,status,home_id,home_team,away_id,away_team,home_goals,away_goals,venue,competition)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(fixture_id) DO UPDATE SET status=excluded.status,home_goals=excluded.home_goals,away_goals=excluded.away_goals,updated_at=CURRENT_TIMESTAMP""",
            (fid, league.get("id"), league.get("season"), parse_match_date(f.get("date")).isoformat(), (f.get("status") or {}).get("short"),
             teams["home"].get("id"), normalize_team_name(teams["home"].get("name")), teams["away"].get("id"), normalize_team_name(teams["away"].get("name")),
             goals.get("home"), goals.get("away"), (f.get("venue") or {}).get("name"), league.get("name")))
        self.conn.commit()

    def upsert_manual_fixture(self, fixture_id: int, date: str, competition: str, home_team: str, away_team: str) -> None:
        self.conn.execute(
            """INSERT INTO fixtures(fixture_id,date,status,home_team,away_team,competition) VALUES (?,?, 'NS',?,?,?)
            ON CONFLICT(fixture_id) DO UPDATE SET date=excluded.date,home_team=excluded.home_team,away_team=excluded.away_team,competition=excluded.competition,updated_at=CURRENT_TIMESTAMP""",
            (int(fixture_id), parse_match_date(date).isoformat(), normalize_team_name(home_team), normalize_team_name(away_team), str(competition).strip()))
        self.conn.commit()

    def complete_manual_fixture(self, fixture_id: int, observed: dict[str, Any]) -> None:
        row = self.conn.execute("SELECT fixture_id,home_team,away_team FROM fixtures WHERE fixture_id=?", (fixture_id,)).fetchone()
        if row is None:
            raise ValueError(f"No existe el fixture_id {fixture_id}.")
        missing = [k for k in STAT_FIELDS if k not in observed]
        if missing:
            raise ValueError(f"Faltan valores observados: {missing}")
        self.conn.execute("UPDATE fixtures SET status='FT',home_goals=?,away_goals=?,updated_at=CURRENT_TIMESTAMP WHERE fixture_id=?", (float(observed["home_goals"]), float(observed["away_goals"]), fixture_id))
        payload = [
            {"team":{"name":row["home_team"]},"statistics":[{"type":"Corner Kicks","value":float(observed["home_corners"])},{"type":"Yellow Cards","value":float(observed["home_cards"])},{"type":"Red Cards","value":0}]},
            {"team":{"name":row["away_team"]},"statistics":[{"type":"Corner Kicks","value":float(observed["away_corners"])},{"type":"Yellow Cards","value":float(observed["away_cards"])},{"type":"Red Cards","value":0}]},
        ]
        self.save_match_payload(fixture_id, "statistics", payload)

    def save_raw(self, endpoint: str, object_id: str | int | None, payload: Any) -> None:
        self.conn.execute("INSERT INTO raw_api(endpoint,object_id,payload_json) VALUES (?,?,?)", (endpoint, str(object_id) if object_id is not None else None, json.dumps(payload, ensure_ascii=False))); self.conn.commit()

    def save_match_payload(self, fixture_id: int, kind: str, payload: Any) -> None:
        table = {"statistics":"match_stats","lineups":"lineups","events":"events"}[kind]
        self.conn.execute(f"INSERT INTO {table}(fixture_id,payload_json) VALUES (?,?) ON CONFLICT(fixture_id) DO UPDATE SET payload_json=excluded.payload_json,updated_at=CURRENT_TIMESTAMP", (fixture_id, json.dumps(payload, ensure_ascii=False))); self.conn.commit()

    def save_prediction(self, fixture_id: int, prediction: dict[str, Any]) -> None:
        model_version = str(prediction.get("metadata", {}).get("model_version", "manual-v0.4"))
        self.conn.execute("INSERT INTO prediction_audits(fixture_id,prediction_json,model_version) VALUES (?,?,?) ON CONFLICT(fixture_id) DO UPDATE SET prediction_json=excluded.prediction_json,model_version=excluded.model_version", (int(fixture_id), json.dumps(prediction, ensure_ascii=False), model_version)); self.conn.commit()

    def get_prediction(self, fixture_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT prediction_json FROM prediction_audits WHERE fixture_id=?", (int(fixture_id),)).fetchone()
        return json.loads(row["prediction_json"]) if row else None

    def update_prediction_observation(self, fixture_id: int, observed: dict[str, Any]) -> dict[str, Any]:
        row = self.conn.execute("SELECT prediction_json FROM prediction_audits WHERE fixture_id=?", (int(fixture_id),)).fetchone()
        if row is None:
            raise ValueError(f"No hay una predicción guardada para fixture_id {fixture_id}.")
        prediction = json.loads(row["prediction_json"]); expected = prediction.get("expected_values", {})
        error = {k: float(observed[k]) - float(expected[k]) for k in STAT_FIELDS if k in expected and k in observed}
        self.conn.execute("UPDATE prediction_audits SET observed_json=?,error_json=?,observed_at=CURRENT_TIMESTAMP WHERE fixture_id=?", (json.dumps(observed), json.dumps(error), int(fixture_id))); self.conn.commit()
        return {"fixture_id": int(fixture_id), "error": error, "prediction": prediction, "observed": observed}

    def historical_dataframe(self, competition: str | None = None) -> pd.DataFrame:
        params = []; where = "WHERE f.status IN ('FT','AET','PEN') AND f.home_goals IS NOT NULL AND f.away_goals IS NOT NULL"
        if competition and competition.lower() != "all":
            where += " AND f.competition=?"; params.append(competition)
        rows = self.conn.execute(f"SELECT f.fixture_id,f.date,f.home_team,f.away_team,f.home_goals,f.away_goals,f.competition,s.payload_json FROM fixtures f LEFT JOIN match_stats s ON s.fixture_id=f.fixture_id {where} ORDER BY f.date", params).fetchall()
        out=[]
        for r in rows:
            home, away = normalize_team_name(r["home_team"]), normalize_team_name(r["away_team"])
            stats = json.loads(r["payload_json"]) if r["payload_json"] else []
            values={"home_corners":None,"away_corners":None,"home_cards":None,"away_cards":None}
            for block in stats if isinstance(stats,list) else []:
                side=normalize_team_name(block.get("team",{}).get("name",""))
                parsed={x.get("type"):self._stat_value(x,"value") for x in block.get("statistics",[])}
                if side==home:
                    values["home_corners"]=parsed.get("Corner Kicks"); values["home_cards"]=(parsed.get("Yellow Cards") or 0)+(parsed.get("Red Cards") or 0)
                elif side==away:
                    values["away_corners"]=parsed.get("Corner Kicks"); values["away_cards"]=(parsed.get("Yellow Cards") or 0)+(parsed.get("Red Cards") or 0)
            if all(values[k] is not None for k in values):
                out.append({"fixture_id":r["fixture_id"],"date":parse_match_date(r["date"]).isoformat(),"home_team":home,"away_team":away,"home_goals":r["home_goals"],"away_goals":r["away_goals"],"competition":r["competition"],**values})
        return pd.DataFrame(out)

    def team_form(self, team: str, n: int = 5, competition: str | None = None) -> dict[str, Any]:
        team=normalize_team_name(team); df=self.historical_dataframe(competition)
        if df.empty: return {"team":team,"matches":0}
        subset=df[(df.home_team==team)|(df.away_team==team)].tail(n).copy()
        if subset.empty: return {"team":team,"matches":0}
        gf=subset.apply(lambda r:r.home_goals if r.home_team==team else r.away_goals,axis=1); ga=subset.apply(lambda r:r.away_goals if r.home_team==team else r.home_goals,axis=1)
        points=[]
        for _,r in subset.iterrows():
            a=r.home_goals if r.home_team==team else r.away_goals; b=r.away_goals if r.home_team==team else r.home_goals; points.append(3 if a>b else 1 if a==b else 0)
        return {"team":team,"matches":int(len(subset)),"points":int(sum(points)),"goals_for_avg":float(gf.mean()),"goals_against_avg":float(ga.mean()),"last_results":["W" if p==3 else "D" if p==1 else "L" for p in points]}

    def learning_bias(self, competition: str, before_date: Any | None = None, min_matches: int = 5, max_matches: int = 80, learning_rate: float = 0.35) -> tuple[dict[str,float], dict[str,Any]]:
        """Learn persistent residual corrections from completed forecasts only.

        Error = observed - predicted. Positive means the model has been
        systematically too low. Corrections are clipped and shrunk so a short
        streak cannot destabilize the model. Only audits whose observed match
        date is before the next prediction are eligible (no leakage).
        """
        params=[competition]; date_clause=""
        if before_date is not None:
            date_clause=" AND f.date < ?"; params.append(parse_match_date(before_date).isoformat())
        rows=self.conn.execute(f"SELECT a.error_json FROM prediction_audits a JOIN fixtures f ON f.fixture_id=a.fixture_id WHERE f.competition=? AND a.observed_json IS NOT NULL AND a.error_json IS NOT NULL {date_clause} ORDER BY f.date DESC LIMIT ?", params+[max_matches]).fetchall()
        errors=[]
        for row in rows:
            try: errors.append(json.loads(row["error_json"]))
            except json.JSONDecodeError: pass
        if len(errors)<min_matches:
            return {k:0.0 for k in STAT_FIELDS},{"status":"insufficient_data","matches":len(errors)}
        frame=pd.DataFrame(errors)
        bias={}
        for k in STAT_FIELDS:
            if k not in frame: bias[k]=0.0; continue
            raw=float(frame[k].mean());
            # Limit learned correction to avoid runaway feedback.
            limit=0.75 if "goals" in k else 1.25 if "corners" in k else 0.90
            bias[k]=float(max(-limit,min(limit,raw*learning_rate)))
        return bias,{"status":"ok","matches":len(errors),"learning_rate":learning_rate}

    def learning_report(self, competition: str | None = None) -> dict[str,Any]:
        clauses=["a.observed_json IS NOT NULL","a.error_json IS NOT NULL"]; params=[]
        if competition and competition.lower()!="all": clauses.append("f.competition=?"); params.append(competition)
        rows=self.conn.execute(f"SELECT a.error_json FROM prediction_audits a JOIN fixtures f ON f.fixture_id=a.fixture_id WHERE {' AND '.join(clauses)}",params).fetchall()
        errors=[]
        for r in rows:
            try: errors.append(json.loads(r["error_json"]))
            except Exception: pass
        if not errors: return {"status":"insufficient_data","matches":0}
        frame=pd.DataFrame(errors)
        return {"status":"ok","matches":len(errors),"mae":{k:float(frame[k].abs().mean()) for k in STAT_FIELDS if k in frame},"bias":{k:float(frame[k].mean()) for k in STAT_FIELDS if k in frame}}

    def close(self) -> None:
        self.conn.close()
