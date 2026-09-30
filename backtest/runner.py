from __future__ import annotations

import pandas as pd
import numpy as np

from football_quant import (
    FootballQuantEngine,
    MatchInput,
)


class BacktestRunner:

    def __init__(
        self,
        simulations: int = 20000,
        bootstrap_runs: int = 100,
        min_train_matches: int = 8,
        seed: int = 42,
        stress_pct: float = 0.10,
    ):
        self.simulations = simulations
        self.bootstrap_runs = bootstrap_runs
        self.min_train_matches = min_train_matches
        self.seed = seed
        self.stress_pct = stress_pct

    # ============================================================
    # 1. CARGAR Y VALIDAR DATOS
    # ============================================================

    def load_data(self, csv_path: str) -> pd.DataFrame:

        df = pd.read_csv(csv_path)

        required_columns = [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_corners",
            "away_corners",
            "home_cards",
            "away_cards",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Faltan columnas en el CSV: {missing}"
            )

        # Fecha
        df["date"] = pd.to_datetime(
            df["date"],
            errors="raise"
        )

        # Estadísticas numéricas
        numeric_columns = [
            "home_goals",
            "away_goals",
            "home_corners",
            "away_corners",
            "home_cards",
            "away_cards",
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise"
            )

        # No permitir valores negativos
        for column in numeric_columns:

            if (df[column] < 0).any():

                raise ValueError(
                    f"Hay valores negativos en {column}"
                )

        # Orden estrictamente cronológico
        df = (
            df
            .sort_values("date")
            .reset_index(drop=True)
        )

        return df

    # ============================================================
    # 2. CALCULAR LEAGUE PRIOR
    # ============================================================

    def calculate_league_prior(
        self,
        history: pd.DataFrame,
    ) -> dict[str, float]:

        """
        Calcula los priors de liga exclusivamente con los
        partidos disponibles ANTES del partido objetivo.

        Esto es fundamental para evitar data leakage.
        """

        required = [
            "home_goals",
            "away_goals",
            "home_corners",
            "away_corners",
            "home_cards",
            "away_cards",
        ]

        prior = {}

        for column in required:

            values = pd.to_numeric(
                history[column],
                errors="coerce"
            )

            mean_value = values.mean()

            if not np.isfinite(mean_value):

                raise ValueError(
                    f"No se pudo calcular league_prior "
                    f"para {column}"
                )

            prior[column] = float(
                mean_value
            )

        return prior

    # ============================================================
    # 3. COMPROBAR SI LOS DOS EQUIPOS TIENEN HISTORIAL
    # ============================================================

    def team_has_history(
        self,
        history: pd.DataFrame,
        team: str,
    ) -> bool:

        home_count = (
            history["home_team"] == team
        ).sum()

        away_count = (
            history["away_team"] == team
        ).sum()

        return (home_count + away_count) > 0

    # ============================================================
    # 4. EXTRAER EXPECTATIVAS
    # ============================================================

    def extract_expectations(
        self,
        engine_result,
    ) -> dict:

        if not hasattr(
            engine_result,
            "expected_values"
        ):

            raise AttributeError(
                "EngineResult no contiene "
                "'expected_values'."
            )

        expected = (
            engine_result.expected_values
        )

        required = [
            "home_goals",
            "away_goals",
            "home_corners",
            "away_corners",
            "home_cards",
            "away_cards",
        ]

        missing = [
            key
            for key in required
            if key not in expected
        ]

        if missing:

            raise KeyError(
                "Faltan valores esperados en "
                f"EngineResult: {missing}\n"
                f"Valores recibidos: {expected}"
            )

        home_goals = float(
            expected["home_goals"]
        )

        away_goals = float(
            expected["away_goals"]
        )

        home_corners = float(
            expected["home_corners"]
        )

        away_corners = float(
            expected["away_corners"]
        )

        home_cards = float(
            expected["home_cards"]
        )

        away_cards = float(
            expected["away_cards"]
        )

        return {

            "pred_home_goals":
                home_goals,

            "pred_away_goals":
                away_goals,

            "pred_total_goals":
                home_goals + away_goals,

            "pred_home_corners":
                home_corners,

            "pred_away_corners":
                away_corners,

            "pred_total_corners":
                home_corners + away_corners,

            "pred_home_cards":
                home_cards,

            "pred_away_cards":
                away_cards,

            "pred_total_cards":
                home_cards + away_cards,
        }

    # ============================================================
    # 5. EXTRAER PROBABILIDADES DE MERCADO
    # ============================================================

    def extract_market_probabilities(
        self,
        engine_result,
    ) -> dict:

        table = engine_result.market_table

        if table is None or table.empty:

            return {}

        output = {}

        for _, row in table.iterrows():

            tier = row.get(
                "tier",
                None
            )

            market = row.get(
                "market",
                None
            )

            line = row.get(
                "line",
                None
            )

            side = row.get(
                "side",
                None
            )

            probability = row.get(
                "probability",
                None
            )

            if (
                market is None
                or line is None
                or side is None
                or probability is None
            ):
                continue

            key = (
                f"{market}|"
                f"{side}|"
                f"{line}"
            )

            output[key] = {

                "probability":
                    float(probability),

                "tier":
                    tier,

                "robust_score":
                    float(
                        row.get(
                            "robust_score",
                            np.nan
                        )
                    ),
            }

        return output

    # ============================================================
    # 6. CALCULAR ERRORES
    # ============================================================

    def calculate_match_errors(
        self,
        result: dict,
    ) -> dict:

        result["goal_abs_error"] = abs(
            result["pred_total_goals"]
            -
            result["actual_total_goals"]
        )

        result["corner_abs_error"] = abs(
            result["pred_total_corners"]
            -
            result["actual_total_corners"]
        )

        result["card_abs_error"] = abs(
            result["pred_total_cards"]
            -
            result["actual_total_cards"]
        )

        return result

    # ============================================================
    # 7. WALK-FORWARD
    # ============================================================

    def run_walk_forward(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        results = []

        total_matches = len(df)

        print()
        print("=" * 75)
        print("FOOTBALL QUANT ENGINE")
        print("WALK-FORWARD BACKTEST")
        print("=" * 75)

        print(
            f"Total de partidos: {total_matches}"
        )

        print(
            f"Mínimo histórico: "
            f"{self.min_train_matches}"
        )

        print()

        # --------------------------------------------------------
        # Recorremos cronológicamente
        # --------------------------------------------------------

        for i in range(total_matches):

            history = (
                df
                .iloc[:i]
                .copy()
            )

            match = df.iloc[i]

            # ----------------------------------------------------
            # Historial mínimo
            # ----------------------------------------------------

            if (
                len(history)
                < self.min_train_matches
            ):

                print(
                    f"[{i + 1}/{total_matches}] "
                    f"OMITIDO | "
                    f"historial = {len(history)}"
                )

                continue

            home_team = (
                match["home_team"]
            )

            away_team = (
                match["away_team"]
            )

            match_date = (
                match["date"]
            )

            print()
            print("-" * 75)

            print(
                f"[{i + 1}/{total_matches}] "
                f"{home_team} vs {away_team}"
            )

            print(
                f"Fecha: {match_date.date()}"
            )

            print(
                f"Historial: "
                f"{len(history)} partidos"
            )

            # ----------------------------------------------------
            # Comprobar historial de equipos
            # ----------------------------------------------------

            home_known = (
                self.team_has_history(
                    history,
                    home_team
                )
            )

            away_known = (
                self.team_has_history(
                    history,
                    away_team
                )
            )

            if not home_known:

                print(
                    f"OMITIDO: "
                    f"{home_team} no tiene historial previo."
                )

                continue

            if not away_known:

                print(
                    f"OMITIDO: "
                    f"{away_team} no tiene historial previo."
                )

                continue

            # ----------------------------------------------------
            # League prior
            # ----------------------------------------------------

            try:

                league_prior = (
                    self.calculate_league_prior(
                        history
                    )
                )

            except Exception as error:

                print()
                print("=" * 75)
                print("ERROR CALCULANDO LEAGUE PRIOR")
                print("=" * 75)

                print(
                    f"Tipo: "
                    f"{type(error).__name__}"
                )

                print(
                    f"Mensaje: {error}"
                )

                print("=" * 75)

                raise

            print()
            print("LEAGUE PRIOR")

            for key, value in (
                league_prior.items()
            ):

                print(
                    f"{key:20s}: "
                    f"{value:.4f}"
                )

            # ----------------------------------------------------
            # Crear motor
            # ----------------------------------------------------

            try:

                engine = FootballQuantEngine(
                    league_prior=league_prior
                )

            except Exception as error:

                print()
                print("=" * 75)
                print("ERROR CREANDO FOOTBALL QUANT ENGINE")
                print("=" * 75)

                print(
                    f"Tipo: "
                    f"{type(error).__name__}"
                )

                print(
                    f"Mensaje: {error}"
                )

                print("=" * 75)

                raise

            # ----------------------------------------------------
            # MatchInput
            # ----------------------------------------------------

            target = MatchInput(
                home_team=home_team,
                away_team=away_team,
            )

            # ----------------------------------------------------
            # Ejecutar fit_predict
            # ----------------------------------------------------

            try:

                engine_result = (
                    engine.fit_predict(
                        history=history,
                        match=target,
                        n_simulations=self.simulations,
                        n_bootstrap=self.bootstrap_runs,
                        seed=self.seed + i,
                        stress_pct=self.stress_pct,
                    )
                )

            except Exception as error:

                print()
                print("=" * 75)
                print("ERROR REAL EN FIT_PREDICT")
                print("=" * 75)

                print(
                    f"Partido: "
                    f"{home_team} vs {away_team}"
                )

                print(
                    f"Tipo: "
                    f"{type(error).__name__}"
                )

                print(
                    f"Mensaje: {error}"
                )

                print("=" * 75)

                raise

            # ----------------------------------------------------
            # Expected values
            # ----------------------------------------------------

            try:

                predicted = (
                    self.extract_expectations(
                        engine_result
                    )
                )

            except Exception as error:

                print()
                print("=" * 75)
                print("ERROR EXTRAYENDO EXPECTATIVAS")
                print("=" * 75)

                print(
                    f"Tipo: "
                    f"{type(error).__name__}"
                )

                print(
                    f"Mensaje: {error}"
                )

                print()
                print(
                    "expected_values recibido:"
                )

                print(
                    getattr(
                        engine_result,
                        "expected_values",
                        None
                    )
                )

                print("=" * 75)

                raise

            # ----------------------------------------------------
            # Market probabilities
            # ----------------------------------------------------

            market_probabilities = (
                self.extract_market_probabilities(
                    engine_result
                )
            )

            # ----------------------------------------------------
            # Resultados reales
            # ----------------------------------------------------

            actual_home_goals = float(
                match["home_goals"]
            )

            actual_away_goals = float(
                match["away_goals"]
            )

            actual_home_corners = float(
                match["home_corners"]
            )

            actual_away_corners = float(
                match["away_corners"]
            )

            actual_home_cards = float(
                match["home_cards"]
            )

            actual_away_cards = float(
                match["away_cards"]
            )

            actual_total_goals = (
                actual_home_goals
                +
                actual_away_goals
            )

            actual_total_corners = (
                actual_home_corners
                +
                actual_away_corners
            )

            actual_total_cards = (
                actual_home_cards
                +
                actual_away_cards
            )

            # ----------------------------------------------------
            # Construir resultado
            # ----------------------------------------------------

            result = {

                "match_number":
                    i + 1,

                "date":
                    match_date,

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "training_matches":
                    len(history),

                # PREDICCIONES
                **predicted,

                # RESULTADOS
                "actual_home_goals":
                    actual_home_goals,

                "actual_away_goals":
                    actual_away_goals,

                "actual_total_goals":
                    actual_total_goals,

                "actual_home_corners":
                    actual_home_corners,

                "actual_away_corners":
                    actual_away_corners,

                "actual_total_corners":
                    actual_total_corners,

                "actual_home_cards":
                    actual_home_cards,

                "actual_away_cards":
                    actual_away_cards,

                "actual_total_cards":
                    actual_total_cards,

                "data_quality":
                    float(
                        engine_result.metadata.get(
                            "data_quality",
                            np.nan
                        )
                    ),
            }

            # ----------------------------------------------------
            # Añadir probabilidades principales
            # ----------------------------------------------------

            for key, value in (
                market_probabilities.items()
            ):

                safe_key = (
                    "market__"
                    +
                    key.replace(
                        "|",
                        "__"
                    ).replace(
                        " ",
                        "_"
                    )
                )

                result[
                    safe_key
                ] = value["probability"]

            # ----------------------------------------------------
            # Errores
            # ----------------------------------------------------

            result = (
                self.calculate_match_errors(
                    result
                )
            )

            results.append(
                result
            )

            # ----------------------------------------------------
            # Mostrar resultado
            # ----------------------------------------------------

            print()
            print("PREDICCIÓN")

            print(
                f"Goles:    "
                f"{predicted['pred_total_goals']:.3f}"
            )

            print(
                f"Corners:  "
                f"{predicted['pred_total_corners']:.3f}"
            )

            print(
                f"Tarjetas: "
                f"{predicted['pred_total_cards']:.3f}"
            )

            print()
            print("REAL")

            print(
                f"Goles:    "
                f"{actual_total_goals:.0f}"
            )

            print(
                f"Corners:  "
                f"{actual_total_corners:.0f}"
            )

            print(
                f"Tarjetas: "
                f"{actual_total_cards:.0f}"
            )

            print()
            print("ERROR ABSOLUTO")

            print(
                f"Goles:    "
                f"{result['goal_abs_error']:.3f}"
            )

            print(
                f"Corners:  "
                f"{result['corner_abs_error']:.3f}"
            )

            print(
                f"Tarjetas: "
                f"{result['card_abs_error']:.3f}"
            )

            print()
            print(
                f"Calidad de datos: "
                f"{result['data_quality']:.3f}"
            )

        # ========================================================
        # DATAFRAME
        # ========================================================

        if not results:

            print()
            print("=" * 75)
            print("NO SE GENERARON PREDICCIONES")
            print("=" * 75)

            return pd.DataFrame()

        return pd.DataFrame(
            results
        )

    # ============================================================
    # 8. RESUMEN
    # ============================================================

    def generate_summary(
        self,
        results: pd.DataFrame
    ):

        if results.empty:

            return {}

        # --------------------------------------------------------
        # Errores
        # --------------------------------------------------------

        goal_error = (
            results["pred_total_goals"]
            -
            results["actual_total_goals"]
        )

        corner_error = (
            results["pred_total_corners"]
            -
            results["actual_total_corners"]
        )

        card_error = (
            results["pred_total_cards"]
            -
            results["actual_total_cards"]
        )

        # --------------------------------------------------------
        # MAE
        # --------------------------------------------------------

        goal_mae = (
            np.abs(goal_error).mean()
        )

        corner_mae = (
            np.abs(corner_error).mean()
        )

        card_mae = (
            np.abs(card_error).mean()
        )

        # --------------------------------------------------------
        # RMSE
        # --------------------------------------------------------

        goal_rmse = np.sqrt(
            np.mean(
                goal_error ** 2
            )
        )

        corner_rmse = np.sqrt(
            np.mean(
                corner_error ** 2
            )
        )

        card_rmse = np.sqrt(
            np.mean(
                card_error ** 2
            )
        )

        # --------------------------------------------------------
        # BIAS
        # --------------------------------------------------------

        goal_bias = (
            goal_error.mean()
        )

        corner_bias = (
            corner_error.mean()
        )

        card_bias = (
            card_error.mean()
        )

        return {

            "predicted_matches":
                len(results),

            "goals": {

                "MAE":
                    float(goal_mae),

                "RMSE":
                    float(goal_rmse),

                "Bias":
                    float(goal_bias),
            },

            "corners": {

                "MAE":
                    float(corner_mae),

                "RMSE":
                    float(corner_rmse),

                "Bias":
                    float(corner_bias),
            },

            "cards": {

                "MAE":
                    float(card_mae),

                "RMSE":
                    float(card_rmse),

                "Bias":
                    float(card_bias),
            },
        }


# =================================================================
# EJECUCIÓN DIRECTA
# =================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------
    # IMPORTANTE:
    #
    # El engine.py exige mínimo 8 partidos históricos.
    #
    # Esto sigue siendo solamente un test técnico porque el
    # sample_matches.csv tiene 16 partidos y muchos equipos aparecen
    # por primera vez justo cuando intentamos predecirlos.
    # -------------------------------------------------------------

    runner = BacktestRunner(

        simulations=5000,

        bootstrap_runs=50,

        min_train_matches=8,

        seed=42,

        stress_pct=0.10,
    )

    # -------------------------------------------------------------
    # Cargar datos
    # -------------------------------------------------------------

    data = runner.load_data(
        "data/sample_matches.csv"
    )

    print()
    print("=" * 75)
    print("DATOS CARGADOS")
    print("=" * 75)

    print(
        f"Partidos: "
        f"{len(data)}"
    )

    print(
        f"Fecha inicial: "
        f"{data['date'].min()}"
    )

    print(
        f"Fecha final: "
        f"{data['date'].max()}"
    )

    # -------------------------------------------------------------
    # Equipos
    # -------------------------------------------------------------

    teams = sorted(
        set(data["home_team"])
        |
        set(data["away_team"])
    )

    print()
    print(
        f"Equipos encontrados: "
        f"{len(teams)}"
    )

    for team in teams:

        print(
            f"  - {team}"
        )

    # -------------------------------------------------------------
    # EJECUTAR
    # -------------------------------------------------------------

    results = runner.run_walk_forward(
        data
    )

    # -------------------------------------------------------------
    # RESUMEN
    # -------------------------------------------------------------

    if not results.empty:

        print()
        print("=" * 75)
        print("RESUMEN DEL BACKTEST")
        print("=" * 75)

        summary = (
            runner.generate_summary(
                results
            )
        )

        print()

        print(
            f"Partidos predichos: "
            f"{summary['predicted_matches']}"
        )

        print()
        print("GOLES")

        print(
            f"MAE:  "
            f"{summary['goals']['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{summary['goals']['RMSE']:.4f}"
        )

        print(
            f"Bias: "
            f"{summary['goals']['Bias']:.4f}"
        )

        print()
        print("CORNERS")

        print(
            f"MAE:  "
            f"{summary['corners']['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{summary['corners']['RMSE']:.4f}"
        )

        print(
            f"Bias: "
            f"{summary['corners']['Bias']:.4f}"
        )

        print()
        print("TARJETAS")

        print(
            f"MAE:  "
            f"{summary['cards']['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{summary['cards']['RMSE']:.4f}"
        )

        print(
            f"Bias: "
            f"{summary['cards']['Bias']:.4f}"
        )

        # ---------------------------------------------------------
        # Guardar CSV
        # ---------------------------------------------------------

        output_file = (
            "data/backtest_results.csv"
        )

        results.to_csv(
            output_file,
            index=False
        )

        print()
        print(
            f"Resultados guardados en:"
        )

        print(
            output_file
        )

        # ---------------------------------------------------------
        # Tabla resumida
        # ---------------------------------------------------------

        print()
        print("=" * 75)
        print("RESULTADOS POR PARTIDO")
        print("=" * 75)

        display_columns = [
            "date",
            "home_team",
            "away_team",

            "pred_total_goals",
            "actual_total_goals",
            "goal_abs_error",

            "pred_total_corners",
            "actual_total_corners",
            "corner_abs_error",

            "pred_total_cards",
            "actual_total_cards",
            "card_abs_error",

            "data_quality",
        ]

        print(
            results[
                display_columns
            ].to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "El runner terminó sin predicciones."
        )