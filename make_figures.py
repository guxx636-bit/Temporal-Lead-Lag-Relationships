"""
生成论文Figure 1和Figure 2
Figure 1: 五个话题的时间序列图
Figure 2: 峰值事件分类柱状图

运行: python3 make_figures.py
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
import matplotlib.dates as mdates

WIKI_DIR = "wiki_esports_data"
GDELT_DIR = "gdelt_esports_data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

TOPICS = [
    ("esports", "Esports"),
    ("video_game_addiction", "Gaming Addiction"),
    ("loot_box", "Loot Box"),
    ("screen_time", "Screen Time"),
    ("video_game_violence", "Game Violence"),
]

# ==============================================
# Figure 1: 时间序列 (5 panels)
# ==============================================

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Figure 1. Daily Time Series of GDELT Mentions and Wikipedia Page Views", fontsize=14, weight="bold")

for idx, (topic, label) in enumerate(TOPICS):
    ax = axes[idx]

    # GDELT
    gdelt_path = os.path.join(GDELT_DIR, f"{topic}.csv")
    if os.path.exists(gdelt_path):
        g = pd.read_csv(gdelt_path, parse_dates=["date"]).sort_values("date")
        ax.plot(g["date"], np.log1p(g["mentions"]), color="#2166AC", lw=0.8, alpha=0.8, label="GDELT")

    # Wikipedia
    wiki_path = os.path.join(WIKI_DIR, f"{topic}.csv")
    if os.path.exists(wiki_path):
        w = pd.read_csv(wiki_path, parse_dates=["date"]).sort_values("date")
        ax2 = ax.twinx()
        ax2.plot(w["date"], np.log1p(w["views"]), color="#E6A817", lw=0.8, alpha=0.8, label="Wikipedia")
        ax2.set_ylabel("Wiki (log views)", color="#E6A817", fontsize=9)
        ax2.tick_params(axis="y", colors="#E6A817")

    ax.set_ylabel("GDELT (log mentions)", color="#2166AC", fontsize=9)
    ax.tick_params(axis="y", colors="#2166AC")
    ax.set_title(label, fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)

axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure1_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close()
print("✅ Figure 1 saved: figures/figure1_timeseries.png")

# ==============================================
# Figure 2: 峰值事件柱状图
# ==============================================

topics_labels = ["Esports", "Gaming\nAddiction", "Screen\nTime", "Game\nViolence", "Loot\nBox"]
media_led = [8, 8, 8, 9, 11]
public_led = [7, 8, 8, 5, 5]
synchronous = [1, 0, 0, 2, 0]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(topics_labels))
width = 0.25

bars1 = ax.bar(x - width, media_led, width, label="Media-leading", color="#2166AC", alpha=0.85)
bars2 = ax.bar(x, public_led, width, label="Public-leading", color="#E6A817", alpha=0.85)
bars3 = ax.bar(x + width, synchronous, width, label="Synchronous", color="#4DAF4A", alpha=0.85)

ax.set_xlabel("Topic", fontsize=11)
ax.set_ylabel("Number of peak events", fontsize=11)
ax.set_title("Figure 2. Classification of Peak Events by Direction", fontsize=13, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(topics_labels, fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(bar.get_height()), ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(bar.get_height()), ha="center", fontsize=9)
for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(bar.get_height()), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure2_peaks.png"), dpi=300, bbox_inches="tight")
plt.close()
print("✅ Figure 2 saved: figures/figure2_peaks.png")

print(f"\n完成！图片已保存至: {FIG_DIR}/")
