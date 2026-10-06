# Temporal Lead-Lag Relationships (GDELT × Wikipedia)

Reproduction code and data for *"Characterizing Lead-Lag Structures Between News Coverage and Public Information-Seeking on Adolescent Digital Health Topics Using GDELT and Wikipedia Page Views."*

## Data

Two daily series per focal topic, covering 2020-01-01 to 2024-12-31:

| Series | Columns | Location |
|---|---|---|
| GDELT news mentions | `date`, `mentions` | `gdelt_esports_data/<topic>.csv` |
| Wikipedia page views | `date`, `views` | `wiki_esports_data/<topic>.csv` |

Focal topics (5): `esports`, `video_game_addiction`, `loot_box`, `screen_time`, `video_game_violence`.

The `<topic>.csv` files at the repository root are the **Wikipedia** series (identical to `wiki_esports_data/`), kept for convenience only — the analysis scripts read the two data subdirectories listed above.

**Sample size (why N = 1,825).** The two series are inner-joined on `date`. The Wikipedia series covers all **1,827** calendar days of the study window; the GDELT 2.0 `timelinevolraw` timeline returned **no records for two days — 2020-10-20 and 2023-03-23** — so the merged series used in all bivariate (Granger) analyses contains **N = 1,825 daily observations**. The descriptive statistics in Table 1 of the manuscript are computed on this merged series.

Ground-truth event annotations: `<topic>_events.json` (5 topics × 16 events = 80), used as the reference events for threshold calibration.

Aggregated results: `granger_results.csv`, `summary.json`. Threshold-calibration outputs: `backtest_pooled.csv`, `backtest_by_topic.csv`.

Extended topics (exploratory only, not part of the five focal topics): `adolescent_depression.csv`, `obesity.csv`, `social_media_health.csv`, `teen_mental_health.csv`, `gaming_disorder.csv`, `social_media_addiction.csv`.

## Dependencies

Python 3.10+. Install with:

```
pip install -r requirements.txt
```

**Version note (statsmodels).** Pin `statsmodels==0.15.0`, as tested. Two API details matter for this repository:

1. `grangercausalitytests` returns, for each lag, a 2-tuple whose first element is a dict keyed by the four test names (`ssr_ftest`, `ssr_chi2test`, `lrtest`, `params_ftest`) and whose second element holds the fitted-model results. `granger_analysis.py` selects the optimal lag by the minimum chi-square p-value, accessed as `gc[lag][0]["ssr_chi2test"][1]`.
2. statsmodels 0.15 **removed the `verbose` argument** from `grangercausalitytests`; the script therefore does not pass it.

## Entry points

- `granger_analysis.py` — Granger causality tests and minimum-p lag selection.
- `backtest_thresholds.py` — threshold-calibration backtest against the annotated peak events.
- `make_figures.py` — time-series and peak-event figures.
- `event_annotation.py` — builds the `*_events.json` ground-truth event annotations.
- `gdelt_one_topic.py` / `wikipedia_esports.py` / `wiki_test.py` — data collection helpers (GDELT `timelinevolraw`, Wikimedia REST API).

## Reproduction

```
pip install -r requirements.txt
python granger_analysis.py                     # -> agenda_setting_results/granger_results.csv
python backtest_thresholds.py . out            # -> out/backtest_pooled.csv, out/backtest_by_topic.csv
```

With the data subdirectories as shipped, `granger_analysis.py` reads both series locally (no API calls are needed). The script's `WIKI_DIR` / `GDELT_DIR` constants expect exactly the two subdirectory names above.
