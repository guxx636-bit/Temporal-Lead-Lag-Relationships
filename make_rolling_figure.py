"""Regenerate Figure 4 (rolling-window Granger stability, loot box public-to-media)
from rolling_window_lootbox.csv — the only reproducible source for these numbers.

Replaces the previous figure, whose annotations (78% significant; optimal lag = 1 day
in 85% of windows; N = 60) could not be reproduced from any pipeline on the frozen
data. Reproducible values (365-day windows, 30-day step, 49 windows):
  - fixed lag-1 significant at P < .05 in 36.7% (18/49) of windows
  - per-window minimum p over lags 1-30 significant in 75.5% (37/49) of windows
  - per-window argmin over lags 1-30 equals 1 day in 10.2% (5/49) of windows

Run:  python make_rolling_figure.py
Out:  figures/rolling_window_lootbox.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

rw = pd.read_csv("rolling_window_lootbox.csv", parse_dates=["window_start", "window_end"])
rw = rw[rw["p_lag1"] != ""].copy()
rw["p_lag1"] = rw["p_lag1"].astype(float)
rw["center"] = rw["window_start"] + (rw["window_end"] - rw["window_start"]) / 2
rw["nlp"] = -np.log10(rw["p_lag1"])

n = len(rw)
sig = int((rw["p_lag1"] < 0.05).sum())
argmin1 = int((rw["argmin_lag_30"].astype(float) == 1).sum())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})

colors = np.where(rw["p_lag1"] < 0.05, "#2e7d32", "#c62828")
ax1.bar(rw["center"], rw["nlp"], width=24, color=colors, edgecolor="white", linewidth=.4)
ax1.axhline(-np.log10(0.05), color="#b71c1c", ls="--", lw=1.2)
ax1.text(rw["center"].iloc[1], -np.log10(0.05) + .07, "P = .05",
         color="#b71c1c", fontsize=9)
ax1.set_ylabel("$-\\log_{10}(P)$, 1-day lag")
ax1.set_ylim(0, max(rw["nlp"].max() * 1.12, 3))
ax1.set_title("Rolling-Window Granger Causality: Loot Box Public to Media (1-day lag)",
              fontsize=12, fontweight="bold")
ax1.annotate(f"{sig}/{n} windows significant at P < .05\n({sig/n:.1%}, fixed 1-day lag)",
             xy=(.985, .55), xycoords="axes fraction", ha="right", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.35", fc="#fff8e1", ec="#999"))

ax2.step(rw["center"], rw["argmin_lag_30"].astype(float), where="mid",
         color="#3b6ea5", lw=1.6)
ax2.set_ylabel("Optimal lag (days)")
ax2.set_xlabel("Window center date")
ax2.set_yticks([1, 5, 10, 15, 20, 25, 30])
ax2.annotate(f"argmin over lags 1\u201330 = 1 day in {argmin1}/{n} windows ({argmin1/n:.1%})",
             xy=(.985, .9), xycoords="axes fraction", ha="right", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.35", fc="#fff8e1", ec="#999"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)

os.makedirs("figures", exist_ok=True)
out = "figures/rolling_window_lootbox.png"
fig.tight_layout()
fig.savefig(out, dpi=200)
print("saved:", out)
print(f"windows={n}  lag-1 significant={sig} ({sig/n:.1%})  argmin==1: {argmin1} ({argmin1/n:.1%})"
      f"  min-p significant: {(rw.p_min_30lags.astype(float)<0.05).sum()} "
      f"({(rw.p_min_30lags.astype(float)<0.05).mean():.1%})")
