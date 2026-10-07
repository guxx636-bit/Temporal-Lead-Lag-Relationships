"""
export_effect_sizes.py — Effect sizes (delta R^2 and RMSE reduction) for the
selected Granger models, on the exact pipeline used for the reported chi-square
statistics (same log1p transform, same ADF-based differencing, same
minimum-p lag selection over lags 1-30, same inner-joined daily series).

Definitions
-----------
For each topic x direction, at the selected lag p:
    restricted   model:  Y_t = a + sum_{k=1..p} b_k Y_{t-k} + e
    unrestricted model:  Y_t = a + sum_{k=1..p} b_k Y_{t-k} + sum_{k=1..p} c_k X_{t-k} + e
    delta R^2    = (SSR_r - SSR_u) / TSS_centered   (incremental variance explained)
    RMSE reduction = 1 - sqrt(SSR_u / SSR_r)        (improvement in prediction accuracy)

Note that with an autocorrelated daily series the two columns are NOT equal to
each other: RMSE reduction = 1 - sqrt(1 - delta_R2 * (TSS / SSR_r)), and
TSS/SSR_r > 1 whenever the restricted (autoregressive) model already explains
part of the variance. Both columns come from the same two fitted models.

Usage:  python export_effect_sizes.py <repo_dir> <out_dir>
Output: effect_size_results.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings("ignore")

TOPICS = [
    ("Esports", "esports"),
    ("Gaming Disorder", "video_game_addiction"),
    ("Loot Box", "loot_box"),
    ("Screen Time", "screen_time"),
    ("Video Game Violence", "video_game_violence"),
]
MAX_LAG = 30
DIRECTIONS = [("Media->Public", "mentions", "wikipedia"), ("Public->Media", "wikipedia", "mentions")]


def build_series(repo_dir, slug):
    """Inner-joined daily log1p series, differenced where ADF says non-stationary."""
    g = pd.read_csv(os.path.join(repo_dir, "gdelt_esports_data", f"{slug}.csv"), parse_dates=["date"])
    w = pd.read_csv(os.path.join(repo_dir, "wiki_esports_data", f"{slug}.csv"), parse_dates=["date"])
    w = w.rename(columns={"views": "wikipedia"})
    d = g.sort_values("date").merge(w.sort_values("date"), on="date")
    for col in ["mentions", "wikipedia"]:
        d[col] = np.log1p(d[col])
        if adfuller(d[col].dropna())[1] > 0.05:
            d[col] = d[col].diff()
    return d.dropna().reset_index(drop=True)


def main(repo_dir, out_dir):
    rows = []
    for topic, slug in TOPICS:
        d = build_series(repo_dir, slug)
        for label, cause, effect in DIRECTIONS:
            gc = grangercausalitytests(d[[effect, cause]], maxlag=MAX_LAG)
            lag, p = min(
                ((l, gc[l][0]["ssr_chi2test"][1]) for l in range(1, MAX_LAG + 1)),
                key=lambda t: t[1],
            )
            chi2 = gc[lag][0]["ssr_chi2test"][0]

            Y = d[effect].to_numpy(float)
            X = d[cause].to_numpy(float)
            n = len(Y)
            mask = np.ones(n, dtype=bool)
            mask[:lag] = False
            y = Y[mask]
            y_lags = [np.roll(Y, k)[mask] for k in range(1, lag + 1)]
            x_lags = [np.roll(X, k)[mask] for k in range(1, lag + 1)]

            res_r = sm.OLS(y, np.column_stack([np.ones(mask.sum())] + y_lags)).fit()
            res_u = sm.OLS(y, np.column_stack([np.ones(mask.sum())] + y_lags + x_lags)).fit()

            ssr_r, ssr_u = res_r.ssr, res_u.ssr
            tss = float(((y - y.mean()) ** 2).sum())
            delta_r2 = (ssr_r - ssr_u) / tss
            rmse_red = 1.0 - np.sqrt(ssr_u / ssr_r)

            rows.append(
                {
                    "topic": topic,
                    "direction": label,
                    "optimal_lag": lag,
                    "chi2": round(chi2, 4),
                    "p_value": p,
                    "n_obs": int(mask.sum()),
                    "delta_r2": round(delta_r2, 4),
                    "rmse_reduction_pct": round(rmse_red * 100, 1),
                }
            )
            print(f"{topic:22s} {label:14s} lag={lag:2d} chi2={chi2:8.2f} dR2={delta_r2:.4f} RMSE_red={rmse_red*100:5.1f}%")

    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "effect_size_results.csv")
    out.to_csv(path, index=False)
    print("\ndelta R^2 range: %.3f-%.3f | RMSE reduction range: %.1f%%-%.1f%%"
          % (out.delta_r2.min(), out.delta_r2.max(),
             out.rmse_reduction_pct.min(), out.rmse_reduction_pct.max()))
    print("wrote", path)
    return out


if __name__ == "__main__":
    repo_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    main(repo_dir, out_dir)
