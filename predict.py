from __future__ import annotations
import argparse, json
import pandas as pd
from football_quant import FootballQuantEngine
from football_quant.automation import SQLiteStore, normalize_team_name
from football_quant.automation.manual import ManualMatchManager

PRIOR={"home_goals":1.45,"away_goals":1.15,"home_corners":5.2,"away_corners":4.4,"home_cards":2.1,"away_cards":2.2}

def main()->None:
    parser=argparse.ArgumentParser(description="Predicción multi-competición con aprendizaje de errores.")
    parser.add_argument("--upcoming",default="data/upcoming_matches.csv"); parser.add_argument("--competition",default="all",help="Filtra por competición; usa all para el CSV completo."); parser.add_argument("--db",default="data/football_quant.db"); parser.add_argument("--simulations",type=int,default=20000); parser.add_argument("--bootstrap",type=int,default=200); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args()
    store=SQLiteStore(args.db)
    try:
        manager=ManualMatchManager(store,FootballQuantEngine(PRIOR)); upcoming=manager.load_upcoming(args.upcoming)
        if args.competition.lower()!="all": upcoming=upcoming[upcoming.competition.str.casefold()==args.competition.casefold()].copy();
        if upcoming.empty: raise SystemExit(f"No hay partidos para competition={args.competition}.")
        temp=args.upcoming if args.competition.lower()=="all" else None
        if temp is None:
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile("w",suffix=".csv",delete=False,encoding="utf-8",newline="") as f: upcoming.to_csv(f,index=False); temp=f.name
        predictions=manager.predict_upcoming(temp,n_simulations=args.simulations,n_bootstrap=args.bootstrap,seed=args.seed)
        for p in predictions:
            print("\n"+"="*72); print(f"{p['home_team']} vs {p['away_team']} — fixture {p['fixture_id']}"); print(f"Fecha: {p['date']} | Competición: {p['competition']}")
            print("EXPECTED VALUES"); [print(f"  {k:18s}: {v:.3f}") for k,v in p["expected_values"].items()]
            print(f"Calidad: {p['metadata'].get('data_quality'):.3f} | Incertidumbre: {p['metadata'].get('uncertainty')} | Histórico: {p['history_matches_used']}")
            print("Aprendizaje:",json.dumps(p['metadata'].get('learning'),ensure_ascii=False)); print("Sesgo aprendido:",json.dumps(p['metadata'].get('learned_bias'),ensure_ascii=False))
            print("Forma:"); print(json.dumps(p['team_form'],ensure_ascii=False,indent=2)); top=pd.DataFrame(p["markets"]).head(5)
            if not top.empty: print(top[["market","line","side","probability","robust_score"]].to_string(index=False))
    finally: store.close()

if __name__=="__main__": main()
