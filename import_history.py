from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from football_quant.automation import SQLiteStore, normalize_team_name
from football_quant.automation.store import parse_match_date

REQUIRED=["date","home_team","away_team","home_goals","away_goals","home_corners","away_corners","home_cards","away_cards"]

def main()->None:
    parser=argparse.ArgumentParser(description="Importa historial multi-competición al almacén SQLite.")
    parser.add_argument("csv",nargs="?",default="data/sample_matches.csv"); parser.add_argument("--db",default="data/football_quant.db"); parser.add_argument("--start-id",type=int,default=500000); parser.add_argument("--competition",default=None,help="Sobrescribe la competición de todas las filas.")
    args=parser.parse_args(); path=Path(args.csv); df=pd.read_csv(path); missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise SystemExit(f"Faltan columnas en {path}: {missing}")
    if args.competition is None and "competition" not in df.columns: raise SystemExit("El CSV debe tener columna competition o debes usar --competition.")
    store=SQLiteStore(args.db)
    try:
        imported=0
        for offset,row in enumerate(df.itertuples(index=False)):
            competition=args.competition if args.competition is not None else str(row.competition).strip()
            if not competition: raise SystemExit(f"Fila {offset+2}: competition vacío.")
            fid=args.start_id+offset; home=normalize_team_name(row.home_team); away=normalize_team_name(row.away_team)
            store.upsert_manual_fixture(fid,parse_match_date(row.date).isoformat(),competition,home,away)
            store.complete_manual_fixture(fid,{k:float(getattr(row,k)) for k in REQUIRED[3:]})
            imported+=1
        print(f"✓ {imported} partidos importados en {args.db}"); print(f"✓ Historial total disponible: {len(store.historical_dataframe())} partidos completos")
    finally: store.close()

if __name__=="__main__": main()