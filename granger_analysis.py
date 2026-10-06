"""
Granger因果分析 — GDELT × Wikipedia
电竞与青少年健康话题的议程设置方向
运行: python3 granger_analysis.py
"""

import requests, json, pandas as pd, numpy as np, time, os
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
warnings.filterwarnings('ignore')

# 配置
TOPICS = [
    {"name": "esports",             "wiki_page": "Esports",                "gdelt_query": "esports"},
    {"name": "video_game_addiction", "wiki_page": "Video_game_addiction",   "gdelt_query": "video game addiction"},
    {"name": "loot_box",             "wiki_page": "Loot_box",               "gdelt_query": "loot box"},
    {"name": "screen_time",          "wiki_page": "Screen_time",            "gdelt_query": "screen time"},
    {"name": "video_game_violence",  "wiki_page": "Video_game_controversies","gdelt_query": "video game violence"},
]

WIKI_DIR = "wiki_esports_data"
GDELT_DIR = "gdelt_esports_data"
OUTPUT_DIR = "agenda_setting_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================
# Step 1: 获取GDELT数据
# ==============================================

def fetch_gdelt(query, start="20200101000000", end="20241231000000", retries=3, topic_name=None):
    """获取GDELT数据—优先读取本地CSV，不存在再从API获取"""
    # 先检查本地是否已有数据
    if topic_name:
        local_path = os.path.join(GDELT_DIR, f"{topic_name}.csv")
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, parse_dates=["date"])
            print(f"  📂 读取本地: {local_path} ({len(df)}天)")
            return df.sort_values("date").reset_index(drop=True)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": query, "mode": "timelinevolraw", "format": "json",
              "STARTDATETIME": start, "ENDDATETIME": end}
    for attempt in range(retries):
        time.sleep(6)
        try:
            r = requests.get(url, params=params, timeout=90)
            text = r.text
            if "Please limit" in text:
                print(f"  限速, 等30秒...")
                time.sleep(30); continue
            data = json.loads(text)
            items = data.get("timeline", [{}])[0].get("data", [])
            if items:
                df = pd.DataFrame([{"date": datetime.strptime(i["date"][:8], "%Y%m%d"), "mentions": int(i["value"])} for i in items])
                df = df.sort_values("date").reset_index(drop=True)
                # 保存到本地
                if topic_name:
                    save_path = os.path.join(GDELT_DIR, f"{topic_name}.csv")
                    df.to_csv(save_path, index=False)
                    print(f"  💾 已保存: {save_path}")
                return df
        except: time.sleep(10)
    return None


# ==============================================
# Step 2: 加载Wikipedia数据
# ==============================================

def load_wikipedia(topic_name):
    path = os.path.join(WIKI_DIR, f"{topic_name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.rename(columns={"views": "wikipedia"}, inplace=True)
    return df


# ==============================================
# Step 3: 平稳性检验 + Granger因果
# ==============================================

def make_stationary(series):
    """确保序列平稳（差分之到平稳为止）"""
    s = series.copy()
    diffs = 0
    while True:
        adf = adfuller(s.dropna())
        if adf[1] < 0.05:
            break
        s = s.diff()
        diffs += 1
        if diffs > 3:
            break
    return s, diffs


# ==============================================
# 主程序
# ==============================================

print("=" * 60)
print("议程设置分析: GDELT × Wikipedia")
print("Granger因果检验")
print("=" * 60)

results = []

for topic in TOPICS:
    print(f"\n--- {topic['name']} ---")

    # 获取GDELT
    print(f"  获取GDELT数据: {topic['gdelt_query']}...")
    gdelt = fetch_gdelt(topic["gdelt_query"], topic_name=topic["name"])
    if gdelt is None or len(gdelt) < 100:
        print(f"  ❌ GDELT数据不足")
        continue

    # 加载Wikipedia
    wiki = load_wikipedia(topic["name"])
    if wiki is None or len(wiki) < 100:
        print(f"  ❌ Wikipedia数据不足")
        continue

    # 合并
    merged = pd.merge(gdelt, wiki, on="date", how="inner")
    print(f"  合并后: {len(merged)}天, {merged['date'].min().date()} 至 {merged['date'].max().date()}")
    print(f"  GDELT: 均值={merged['mentions'].mean():.1f}")
    print(f"  Wiki: 均值={merged['wikipedia'].mean():.1f}")

    # 对数变换+标准化
    merged["m"] = np.log1p(merged["mentions"])
    merged["w"] = np.log1p(merged["wikipedia"])

    # 平稳性
    m_stat, m_d = make_stationary(merged["m"])
    w_stat, w_d = make_stationary(merged["w"])
    print(f"  平稳化: GDELT差{m_d}阶, Wiki差{w_d}阶")

    # Granger因果检验 (媒体→公众)
    data = pd.concat([m_stat, w_stat], axis=1).dropna()
    data.columns = ["media", "public"]
    max_lag = 30

    try:
        # Note: statsmodels 0.15 removed the `verbose` argument from
        # grangercausalitytests, so it is omitted here.
        gc_m2p = grangercausalitytests(data[["public", "media"]], max_lag)
        gc_p2m = grangercausalitytests(data[["media", "public"]], max_lag)

        # 找最优滞后（最小p值）
        best_m2p = min([(lag, gc_m2p[lag][0]["ssr_chi2test"][1]) for lag in range(1, max_lag+1)], key=lambda x: x[1])
        best_p2m = min([(lag, gc_p2m[lag][0]["ssr_chi2test"][1]) for lag in range(1, max_lag+1)], key=lambda x: x[1])

        print(f"  媒体→公众: F={best_m2p[0]}lag, p={best_m2p[1]:.6f} {'✅' if best_m2p[1]<0.05 else '❌'}")
        print(f"  公众→媒体: F={best_p2m[0]}lag, p={best_p2m[1]:.6f} {'✅' if best_p2m[1]<0.05 else '❌'}")

        if best_m2p[1] < 0.05 and best_p2m[1] < 0.05:
            direction = "双向"
        elif best_m2p[1] < 0.05:
            direction = "媒体→公众"
        elif best_p2m[1] < 0.05:
            direction = "公众→媒体"
        else:
            direction = "无显著关联"

        print(f"  议程设置方向: {direction}")

        results.append({
            "topic": topic["name"],
            "gdelt_mean": float(merged["mentions"].mean()),
            "wiki_mean": float(merged["wikipedia"].mean()),
            "correlation": float(merged["m"].corr(merged["w"])),
            "media_to_public_lag": int(best_m2p[0]),
            "media_to_public_p": float(best_m2p[1]),
            "public_to_media_lag": int(best_p2m[0]),
            "public_to_media_p": float(best_p2m[1]),
            "direction": direction,
        })

    except Exception as e:
        print(f"  ❌ Granger检验失败: {e}")

# 汇总
print(f"\n{'='*70}")
print("汇总: 议程设置方向")
print("=" * 70)
print(f"\n{'话题':<20s} {'GDELT均值':<12s} {'Wiki均值':<12s} {'相关性':<10s} {'主导方向':<16s}")
print("-" * 70)
for r in results:
    print(f"{r['topic']:<20s} {r['gdelt_mean']:<12.1f} {r['wiki_mean']:<12.1f} {r['correlation']:<10.3f} {r['direction']:<16s}")

# 保存
df_out = pd.DataFrame(results)
df_out.to_csv(os.path.join(OUTPUT_DIR, "granger_results.csv"), index=False)

# 判断: 如果GDELT数据获取失败，用现有的Wikipedia数据先探索
wiki_files = [f for f in os.listdir(WIKI_DIR) if f.endswith(".csv")]
if not results and wiki_files:
    print(f"\n⚠️ GDELT数据获取受限，但Wikipedia数据已存在({len(wiki_files)}个话题)。")
    print(f"  建议: 先单独分析Wikipedia浏览量趋势，等GDELT限速解除后再补跑完整分析。")

print(f"\n结果: {OUTPUT_DIR}/granger_results.csv")
print("完成!")
