"""
测试Wikipedia API是否可用
运行: python3 wiki_test.py
"""
import requests

# 测试之前成功过的话题
url1 = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Esports/daily/20200101/20200131"
try:
    r = requests.get(url1, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    print(f"Esports: 状态码={r.status_code}")
    if r.status_code == 200:
        print(f"  ✅ 成功: {len(r.json().get('items',[]))}条数据")
    else:
        print(f"  ❌ {r.text[:200]}")
except Exception as e:
    print(f"  ❌ {e}")

# 测试新话题
url2 = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Mental_health/daily/20200101/20200131"
try:
    r = requests.get(url2, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    print(f"Mental_health: 状态码={r.status_code}")
    if r.status_code == 200:
        print(f"  ✅ 成功: {len(r.json().get('items',[]))}条数据")
    else:
        print(f"  ❌ {r.text[:200]}")
except Exception as e:
    print(f"  ❌ {e}")
