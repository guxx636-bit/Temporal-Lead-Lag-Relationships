"""
逐个获取GDELT数据（每次只跑一个话题，避免限速）
运行: python3 gdelt_one_topic.py
"""

import requests, json, pandas as pd, time, os
from datetime import datetime

# 还缺的两个话题
MISSING_TOPICS = [
    {"name": "esports",             "query": "esports",                    "file": "esports.csv"},
    {"name": "video_game_addiction", "query": "video game addiction",     "file": "video_game_addiction.csv"},
]

url = "https://api.gdeltproject.org/api/v2/doc/doc"
SAVE_DIR = "gdelt_esports_data"
os.makedirs(SAVE_DIR, exist_ok=True)

for topic in MISSING_TOPICS:
    print(f"\n等待60秒确保GDELT解封...")
    time.sleep(60)

    print(f"获取: {topic['name']} ({topic['query']})")
    params = {
        "query": topic["query"],
        "mode": "timelinevolraw",
        "format": "json",
        "STARTDATETIME": "20200101000000",
        "ENDDATETIME": "20241231000000",
    }

    try:
        r = requests.get(url, params=params, timeout=60)
        text = r.text

        if "Please limit" in text:
            print(f"  仍限速，等2分钟后重试...")
            time.sleep(120)
            r = requests.get(url, params=params, timeout=60)
            text = r.text

        data = json.loads(text)
        items = data.get("timeline", [{}])[0].get("data", [])

        if items:
            df = pd.DataFrame([{"date": datetime.strptime(i["date"][:8], "%Y%m%d"), "mentions": int(i["value"])} for i in items])
            df = df.sort_values("date").reset_index(drop=True)
            path = os.path.join(SAVE_DIR, topic["file"])
            df.to_csv(path, index=False)
            print(f"  ✅ {len(df)}天, 均值={df['mentions'].mean():.1f}/d, 峰值={df['mentions'].max()}")
        else:
            print(f"  ❌ 无数据")

    except Exception as e:
        print(f"  错误: {e}")

    print(f"  完成，等待2分钟再跑下一个...")
    time.sleep(120)

print(f"\n完成！数据在 {SAVE_DIR}/")
