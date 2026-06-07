"""
扩展话题 — Wikipedia + GDELT对照话题
运行: python3 expand_topics.py
"""

import requests, json, pandas as pd, time, os, numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 新增对照话题
NEW_TOPICS = [
    # 健康对照（非游戏）
    {"name": "teen_mental_health",     "wiki": "Mental_health",                "gdelt": "teen mental health"},
    {"name": "adolescent_depression",  "wiki": "Depression_in_childhood_and_adolescence", "gdelt": "adolescent depression"},
    {"name": "obesity",                "wiki": "Obesity",                      "gdelt": "childhood obesity"},
    # 科技/社会议题
    {"name": "artificial_intelligence","wiki": "Artificial_intelligence",      "gdelt": "artificial intelligence"},
    {"name": "climate_change",         "wiki": "Climate_change",               "gdelt": "climate change"},
    # 数字健康对比
    {"name": "social_media_health",    "wiki": "Social_media_and_mental_health","gdelt": "social media mental health"},
    {"name": "smartphone_addiction",   "wiki": "Smartphone_addiction",         "gdelt": "smartphone addiction"},
    # 外部冲击
    {"name": "covid19",                "wiki": "COVID-19_pandemic",            "gdelt": "COVID-19"},
]

WIKI_DIR = "wiki_esports_data"
GDELT_DIR = "gdelt_esports_data"
os.makedirs(WIKI_DIR, exist_ok=True)
os.makedirs(GDELT_DIR, exist_ok=True)

# ==============================================
# Part 1: Wikipedia数据
# ==============================================

def fetch_wiki(page, start="20200101", end="20241231"):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page}/daily/{start}/{end}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code != 200:
            return None
        items = r.json().get("items", [])
        records = [{"date": datetime.strptime(i["timestamp"][:8], "%Y%m%d"), "views": i["views"]} for i in items]
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df
    except:
        return None

print("=" * 60)
print("Part 1: Wikipedia 扩展话题")
print("=" * 60)

wiki_results = []
for topic in NEW_TOPICS:
    csv_path = os.path.join(WIKI_DIR, f"{topic['name']}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        print(f"  📂 {topic['name']}: 已有本地数据 ({len(df)}天)")
        wiki_results.append({"name": topic["name"], "days": len(df)})
        continue

    print(f"\n  {topic['name']} ({topic['wiki']})...")
    df = fetch_wiki(topic["wiki"])
    if df is not None and len(df) > 100:
        df.to_csv(csv_path, index=False)
        print(f"  ✅ {len(df)}天, 均值={df['views'].mean():.0f}/d, 峰值={df['views'].max():,}")
        wiki_results.append({"name": topic["name"], "days": len(df)})
    else:
        print(f"  ❌ 无数据")
    time.sleep(1)

print(f"\nWikipedia: {len(wiki_results)}/{len(NEW_TOPICS)} 完成")

# ==============================================
# Part 2: GDELT数据（逐个获取，避免限速）
# ==============================================

print(f"\n{'='*60}")
print("Part 2: GDELT 扩展话题")
print("=" * 60)

for topic in NEW_TOPICS:
    csv_path = os.path.join(GDELT_DIR, f"{topic['name']}.csv")
    if os.path.exists(csv_path):
        print(f"  📂 {topic['name']}: 已有本地数据")
        continue

    print(f"\n  {topic['name']} ({topic['gdelt']})...")
    print(f"  等待40秒避免限速...")
    time.sleep(40)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": topic["gdelt"], "mode": "timelinevolraw", "format": "json",
              "STARTDATETIME": "20200101000000", "ENDDATETIME": "20241231000000"}
    try:
        r = requests.get(url, params=params, timeout=60)
        text = r.text
        if "Please limit" in text:
            print(f"  限速,等60秒...")
            time.sleep(60)
            r = requests.get(url, params=params, timeout=60)
            text = r.text
        data = json.loads(text)
        items = data.get("timeline", [{}])[0].get("data", [])
        if items and len(items) > 100:
            df = pd.DataFrame([{"date": datetime.strptime(i["date"][:8], "%Y%m%d"), "mentions": int(i["value"])} for i in items])
            df = df.sort_values("date").reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            print(f"  ✅ {len(df)}天, 均值={df['mentions'].mean():.1f}/d")
        else:
            print(f"  ❌ 数据不足")
    except Exception as e:
        print(f"  错误: {e}")

print(f"\n全部完成！")
print(f"Wikipedia: {WIKI_DIR}/")
print(f"GDELT: {GDELT_DIR}/")
