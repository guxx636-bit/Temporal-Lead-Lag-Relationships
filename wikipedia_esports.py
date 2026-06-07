"""
Wikipedia 页面浏览数据获取 — 电竞与青少年健康
免费，无需注册，格式兼容现有EWS代码

运行: python3 wikipedia_esports.py
"""

import requests, pandas as pd, time, os, json
from datetime import datetime, timedelta

OUTPUT_DIR = "wiki_esports_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 话题页面对应的Wikipedia文章标题
TOPICS = [
    {"name": "video_game_addiction",    "page": "Video_game_addiction",      "label": "Video Game Addiction"},
    {"name": "esports",                  "page": "Esports",                    "label": "Esports"},
    {"name": "loot_box",                 "page": "Loot_box",                  "label": "Loot Box"},
    {"name": "video_game_violence",      "page": "Video_game_controversies",   "label": "Game Violence"},
    {"name": "screen_time",              "page": "Screen_time",                "label": "Screen Time"},
    {"name": "gaming_disorder",          "page": "Internet_gaming_disorder",  "label": "Internet Gaming Disorder"},
    {"name": "social_media_addiction",   "page": "Social_media_addiction",    "label": "Social Media Addiction"},
]

# 时间范围
START = "20200101"
END = "20241231"


def fetch_wikipedia_pageviews(page, start, end):
    """获取单个Wikipedia页面的每日浏览量"""
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page}/daily/{start}/{end}"
    headers = {"User-Agent": "Mozilla/5.0 (research project)"}

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"    状态码 {r.status_code}: {r.text[:100]}")
            return None
        data = r.json()
        items = data.get("items", [])
        if not items:
            print(f"    无数据")
            return None
        records = []
        for item in items:
            d = datetime.strptime(str(item["timestamp"]), "%Y%m%d%H")
            records.append({"date": d, "views": item["views"]})
        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"    错误: {e}")
        return None


print("=" * 60)
print("Wikipedia 页面浏览数据 — 电竞与青少年健康")
print(f"时间: {START} 至 {END}")
print("=" * 60)

all_results = []
for topic in TOPICS:
    print(f"\n{topic['label']} (page: {topic['page']})...")
    df = fetch_wikipedia_pageviews(topic["page"], START, END)

    if df is not None and len(df) > 0:
        # 保存
        csv_path = os.path.join(OUTPUT_DIR, f"{topic['name']}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✅ {len(df)} days, 均值={df['views'].mean():.0f}/d, 峰值={df['views'].max():,}")
        print(f"  保存: {csv_path}")

        # 简单断点检测（PELT）
        try:
            import ruptures as rpt
            import numpy as np
            y = df["views"].values.reshape(-1, 1)
            bps = rpt.Pelt(model="l2").fit(y).predict(pen=2 * np.log(len(y)))
            bp = next((b for b in bps if 30 < b < len(y) - 30), len(y) // 2)
            print(f"  断点: {df.iloc[bp]['date'].date()} (第{bp}天)")
        except:
            pass

        all_results.append({
            "name": topic["name"],
            "label": topic["label"],
            "days": len(df),
            "mean": float(df["views"].mean()),
            "max": int(df["views"].max()),
        })
    else:
        print(f"  ❌ 获取失败")

    time.sleep(1)  # 礼貌性延迟

# 汇总
print(f"\n{'='*60}")
print("汇总")
print("=" * 60)
for r in all_results:
    print(f"  {r['label']:<25s} {r['days']:>5d}d  均值={r['mean']:>8.0f}/d  峰值={r['max']:>8,}")

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n完成！数据保存至: {OUTPUT_DIR}/")
