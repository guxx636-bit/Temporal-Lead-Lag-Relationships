"""
事件标注分析 — 检测每个话题的峰值事件并判断谁先驱动谁
运行: python3 event_annotation.py
"""

import pandas as pd, numpy as np, os, json
from datetime import datetime, timedelta

WIKI_DIR = "wiki_esports_data"
GDELT_DIR = "gdelt_esports_data"
OUTPUT_DIR = "agenda_setting_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 需要分析的5个主话题
TOPICS = [
    {"name": "esports"},
    {"name": "video_game_addiction"},
    {"name": "loot_box"},
    {"name": "screen_time"},
    {"name": "video_game_violence"},
]

def find_peaks(series, window=30, n_peaks=10):
    """用滑动窗口检测峰值"""
    peaks = []
    for i in range(window, len(series) - window):
        if series[i] == np.max(series[i-window:i+window]):
            peaks.append((i, series[i]))
    # 取前n_peaks个最高峰值，按值降序
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:n_peaks]


for topic in TOPICS:
    name = topic["name"]
    wiki_path = os.path.join(WIKI_DIR, f"{name}.csv")
    gdelt_path = os.path.join(GDELT_DIR, f"{name}.csv")

    if not os.path.exists(wiki_path) or not os.path.exists(gdelt_path):
        print(f"  ❌ {name}: 数据缺失")
        continue

    # 读取数据
    wiki = pd.read_csv(wiki_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    gdelt = pd.read_csv(gdelt_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # 合并
    merged = pd.merge(gdelt[["date", "mentions"]], wiki[["date", "views"]], on="date", how="inner")

    # 对数变换
    merged["g"] = np.log1p(merged["mentions"])
    merged["w"] = np.log1p(merged["views"])

    # 检测GDELT的峰值
    g_peaks = find_peaks(merged["g"].values, window=15, n_peaks=8)
    w_peaks = find_peaks(merged["w"].values, window=15, n_peaks=8)

    print(f"\n=== {name} ===")

    # 合并所有峰值并分类
    all_events = []
    for idx, val in g_peaks:
        date = merged.iloc[idx]["date"]
        g_val = merged.iloc[idx]["g"]
        w_val = merged.iloc[idx]["w"]
        # 看前5天GDELT和Wikipedia的趋势
        g_before = merged.iloc[max(0,idx-5):idx]["g"].mean()
        w_before = merged.iloc[max(0,idx-5):idx]["w"].mean()
        if g_before > 0 and w_before > 0:
            g_trend = (g_val - g_before) / g_before
            w_trend = (w_val - w_before) / w_before
            if g_trend > w_trend * 1.2:
                direction = "GDELT领先"
            elif w_trend > g_trend * 1.2:
                direction = "Wiki领先"
            else:
                direction = "同步"
        else:
            direction = "不确定"
        all_events.append({"date": str(date.date()), "type": "GDELT峰值", "direction": direction, "g_value": round(float(g_val),2), "w_value": round(float(w_val),2)})

    for idx, val in w_peaks:
        date = merged.iloc[idx]["date"]
        # 检查这个日期是否已记录过
        if str(date.date()) in [e["date"] for e in all_events]:
            continue
        g_val = merged.iloc[idx]["g"]
        w_val = merged.iloc[idx]["w"]
        g_before = merged.iloc[max(0,idx-5):idx]["g"].mean()
        w_before = merged.iloc[max(0,idx-5):idx]["w"].mean()
        if g_before > 0 and w_before > 0:
            g_trend = (g_val - g_before) / g_before
            w_trend = (w_val - w_before) / w_before
            if w_trend > g_trend * 1.2:
                direction = "Wiki领先"
            elif g_trend > w_trend * 1.2:
                direction = "GDELT领先"
            else:
                direction = "同步"
        else:
            direction = "不确定"
        all_events.append({"date": str(date.date()), "type": "Wiki峰值", "direction": direction, "g_value": round(float(g_val),2), "w_value": round(float(w_val),2)})

    # 按日期排序
    all_events.sort(key=lambda x: x["date"])

    # 统计
    media_led = sum(1 for e in all_events if e["direction"] == "GDELT领先")
    public_led = sum(1 for e in all_events if e["direction"] == "Wiki领先")
    simultaneous = sum(1 for e in all_events if e["direction"] == "同步")

    print(f"  检测到 {len(all_events)} 个事件峰值:")
    print(f"  媒体领先(GDELT→Wiki): {media_led}")
    print(f"  公众领先(Wiki→GDELT): {public_led}")
    print(f"  同步: {simultaneous}")
    print(f"  典型事件:")
    for e in all_events[:5]:
        print(f"    {e['date']} {e['type']} ({e['direction']})")

    # 保存
    with open(os.path.join(OUTPUT_DIR, f"{name}_events.json"), "w") as f:
        json.dump(all_events, f, indent=2)

print(f"\n事件数据保存至: {OUTPUT_DIR}/")
