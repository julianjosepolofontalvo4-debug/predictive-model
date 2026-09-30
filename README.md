# Proyecto: predictive-model (football_quant)

Modelo de predicción estadística de fútbol. Abajo está el árbol de archivos y el contenido completo de cada archivo.

## Estructura

```
README.md
backtest/runner.py
baseline.py
data/history_template_multi_competition.csv
data/sample_matches.csv
data/upcoming_matches.csv
data/upcoming_multi_competition.csv
feedback_loop.py
import_history.py
predict.py
pyproject.toml
record_result.py
run_example.py
scheduler.py
src/football_quant/__init__.py
src/football_quant/automation/__init__.py
src/football_quant/automation/api_football.py
src/football_quant/automation/feedback.py
src/football_quant/automation/manual.py
src/football_quant/automation/store.py
src/football_quant/automation/sync.py
src/football_quant/automation/team_names.py
src/football_quant/core.py
src/football_quant/engine.py
src/football_quant/markets.py
src/football_quant/simulation.py
src/football_quant/uncertainty.py
sync_data.py
tests/test_automation.py
tests/test_engine.py
tests/test_manual_workflow.py
```

## Archivos

### `README.md`

```markdown
# Football Quant Engine v0.4 — Multi-Competition + Error Learning

Motor probabilístico para fútbol diseñado para **LaLiga, Premier League y Champions League** (y extensible a otras competiciones).

## Qué mejora esta versión

- **Aislamiento por competición:** el modelo nunca mezcla automáticamente historial de LaLiga con Premier/Champions.
- **Cero fugas temporales:** para un partido futuro solo usa partidos de esa competición con fecha anterior.
- **Fechas robustas:** acepta ISO con y sin zona horaria mediante un parser centralizado.
- **Nombres robustos:** alias comunes se convierten a nombres canónicos antes de guardar/entrenar.
- **Aprendizaje de errores:** después de registrar un resultado, calcula `observado - predicho`; las desviaciones persistentes se convierten en una corrección pequeña y acotada para futuras predicciones.
- **Aprendizaje conservador:** requiere varias observaciones, aplica shrinkage/clipping y evita que una racha corta destruya el modelo.
- **Forma por competición:** la forma reciente de un equipo se calcula dentro de su competición.
- **Bootstrap de datos:** la incertidumbre de mercados se estima re-muestreando históricos y reajustando parámetros.
- **Snapshots reproducibles:** cada entrenamiento guarda el histórico y una versión hash.
- **Auditoría:** cada predicción queda almacenada y luego vinculada con el resultado real.

> No existe un porcentaje de error mínimo garantizable en fútbol. La arquitectura está pensada para **medir, detectar sesgos, recalibrar y mejorar**, no para prometer aciertos. La literatura de modelado deportivo también destaca que calibración y evaluación temporal son tan importantes como la precisión bruta.

## Estructura recomendada de datos

Historial: `date,competition,home_team,away_team,home_goals,away_goals,home_corners,away_corners,home_cards,away_cards`

Puedes importar cada competición por separado:

```powershell
python import_history.py data/laliga_history.csv --competition LaLiga --start-id 1000000
python import_history.py data/premier_history.csv --competition Premier --start-id 2000000
python import_history.py data/champions_history.csv --competition Champions --start-id 3000000
```

También puedes incluir `competition` dentro del CSV y omitir `--competition`.

## Predicción

```powershell
python predict.py --upcoming data/upcoming_multi_competition.csv --competition LaLiga
python predict.py --upcoming data/upcoming_multi_competition.csv --competition Premier
python predict.py --upcoming data/upcoming_multi_competition.csv --competition Champions
python predict.py --upcoming data/upcoming_multi_competition.csv --competition all
```

`all` procesa cada fila con el historial de su propia competición.

## Registrar el resultado

```powershell
python record_result.py 100002
```

Al registrar el partido, el sistema:
1. guarda el resultado,
2. calcula los errores por métrica,
3. deja el error disponible para aprendizaje posterior.

## Ver aprendizaje

```powershell
python feedback_loop.py
```

## Tests

```powershell
python -m pytest -q
```

## Principio de diseño

El modelo no se "autoentrena" con datos futuros. Cada predicción se congela en el momento de generarse; el resultado solo entra en el aprendizaje después de que el partido termina. Esto evita contaminar retrospectivamente las predicciones.
```