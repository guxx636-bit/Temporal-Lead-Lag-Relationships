"""
export_robustness.py — 导出正文 Robustness 小节所需的全部数字
复刻 granger_analysis.py 流水线（log1p → ADF 差分 → grangercausalitytests 卡方, 最小 p 选滞后），
并与 granger_results.csv 对账。

产出 7 份 CSV：
  robustness_lockdown_removal.csv      剔除 2020-03-01..2020-05-31（92 天）后重估最优滞后
  multivariate_dummies_lagcheck.csv    加入 COVID/政策虚拟变量后的最优滞后与显著性
  multivariate_dummies_coefficients.csv 虚拟变量系数（原始 + 标准化尺度）
  cross_spectral_coherence.csv         Welch 相干谱（256 天窗, Hann）
  tls_events.csv                       逐事件 TLS（原始与对数两种口径）
  tls_summary_by_topic.csv             TLS 分话题汇总
  rolling_window_lootbox.csv           loot box 365 天滚动窗 P→M lag-1 显著性

运行: python export_robustness.py
"""

import json, os, warnings
import numpy as np
import pandas as pd
from scipy.signal import coherence, find_peaks
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

TOPICS = ["esports", "video_game_addiction", "loot_box", "screen_time", "video_game_violence"]
GDELT_DIR, WIKI_DIR = "gdelt_esports_data", "wiki_esports_data"
MAXLAG = 30
LOCKDOWN = ("2020-03-01", "2020-05-31")          # 92 天（31+30+31）
COVID_PERIODS = [("2020-03-01", "2020-05-31"),   # 严格封锁期
                 ("2020-11-01", "2021-01-31"),
                 ("2022-01-01", "2022-02-28")]
POLICY_DUMMIES = {
    "uk_lootbox_call": [("2020-09-01", "2020-09-30")],   # UK DCMS call for evidence
    # 正文口径 "major US mass shootings: event dates"；2020–2024 研究期内的主要事件
    "us_mass_shootings": [("2021-03-16", "2021-03-16"),  # Atlanta spa
                          ("2021-03-22", "2021-03-22"),  # Boulder
                          ("2022-05-14", "2022-05-14"),  # Buffalo
                          ("2022-05-24", "2022-05-24"),  # Uvalde
                          ("2023-10-25", "2023-10-25")], # Lewiston
}
# ICD-11 gaming disorder 自 2019-06 起生效——样本始于 2020-01，步进虚拟变量在样本内恒为 1，
# 与截距共线，自动剔除（脚本会打印说明）。


# ---------- 与 granger_analysis.py 完全一致的流水线 ----------

def make_stationary(series):
    s = series.copy()
    while adfuller(s.dropna())[1] >= 0.05:
        s = s.diff()
    return s

def load_topic(name):
    g = pd.read_csv(f"{GDELT_DIR}/{name}.csv", parse_dates=["date"]).sort_values("date")
    w = pd.read_csv(f"{WIKI_DIR}/{name}.csv", parse_dates=["date"]).sort_values("date").rename(columns={"views": "wikipedia"})
    m = pd.merge(g, w, on="date", how="inner").reset_index(drop=True)
    m["m"] = np.log1p(m["mentions"])
    m["w"] = np.log1p(m["wikipedia"])
    st = pd.concat([make_stationary(m["m"]), make_stationary(m["w"])], axis=1).dropna()
    st.columns = ["media", "public"]
    st = st.merge(m[["date"]], left_index=True, right_index=True, how="left")
    return m, st

def granger_best(data, cols, maxlag=MAXLAG):
    """返回 (最优滞后, p, 全部 p)。cols=[y, x]，检验 x→y。"""
    gc = grangercausalitytests(data[cols], maxlag)
    ps = {L: gc[L][0]["ssr_chi2test"][1] for L in range(1, maxlag + 1)}
    best = min(ps, key=ps.get)
    return best, ps[best], ps


# ---------- 对账：基线必须与 granger_results.csv 一致 ----------

print("== 对账：基线最优滞后 vs granger_results.csv ==")
baseline = {}
gr = pd.read_csv("granger_results.csv").set_index("topic")
for name in TOPICS:
    m, st = load_topic(name)
    l1, p1, _ = granger_best(st, ["public", "media"])   # media→public
    l2, p2, _ = granger_best(st, ["media", "public"])   # public→media
    baseline[name] = {"m2p": (l1, p1), "p2m": (l2, p2), "merged": m, "st": st}
    ok1 = int(l1) == int(gr.loc[name, "media_to_public_lag"])
    ok2 = int(l2) == int(gr.loc[name, "public_to_media_lag"])
    print(f"  {name:<22s} M→P {l1:>2d} (csv {int(gr.loc[name,'media_to_public_lag']):>2d}) {'OK' if ok1 else 'MISMATCH'} | "
          f"P→M {l2:>2d} (csv {int(gr.loc[name,'public_to_media_lag']):>2d}) {'OK' if ok2 else 'MISMATCH'}")


# ---------- A. 剔除封锁期 ----------

print("\n== A. 剔除 2020-03-01..2020-05-31（92 天）==")
rows = []
ld0, ld1 = pd.Timestamp(LOCKDOWN[0]), pd.Timestamp(LOCKDOWN[1])
for name in TOPICS:
    st = baseline[name]["st"]
    kept = st[(st["date"] < ld0) | (st["date"] > ld1)].drop(columns="date").reset_index(drop=True)
    n_drop = len(st) - len(kept)
    for label, cols in [("media_to_public", ["public", "media"]), ("public_to_media", ["media", "public"])]:
        bl, bp, _ = granger_best(st.drop(columns="date"), cols)
        el, ep, _ = granger_best(kept, cols)
        rows.append({"topic": name, "direction": label, "n_dropped": n_drop,
                     "baseline_lag": bl, "baseline_p": f"{bp:.3e}",
                     "excl_lag": el, "excl_p": f"{ep:.3e}",
                     "lag_change_days": int(el - bl), "excl_significant": ep < 0.01})
lock = pd.DataFrame(rows)
lock.to_csv("robustness_lockdown_removal.csv", index=False)
print(lock.to_string(index=False))
print("最大滞后变动:", lock.lag_change_days.abs().max(),
      "| 剔除后全部显著(P<.01):", bool((lock.excl_p.astype(float) < 0.01).all()))


# ---------- B. 多元 Granger + 虚拟变量 ----------

print("\n== B. 多元 Granger（含 COVID/政策虚拟变量）==")

def build_dummies(dates, topic):
    d = pd.DataFrame({"date": dates})
    for i, (a, b) in enumerate(COVID_PERIODS, 1):
        d[f"covid_{i}"] = ((d.date >= a) & (d.date <= b)).astype(float)
    for k, periods in POLICY_DUMMIES.items():
        if k == "uk_lootbox_call" and topic != "loot_box":
            continue
        if k == "us_mass_shootings" and topic != "video_game_violence":
            continue
        v = pd.Series(0.0, index=d.index)
        for a, b in periods:
            v += ((d.date >= a) & (d.date <= b)).astype(float)
        d[k] = (v > 0).astype(float)
    return d

def mv_fit(y, X_exog_lags, D, cov="nonrobust"):
    """y: 因变量序列; X_exog_lags: DataFrame(预测变量滞后项); D: 虚拟变量 DataFrame(无 date)"""
    X = pd.concat([X_exog_lags, D], axis=1)
    X = sm.add_constant(X)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7}) if cov == "hac" else OLS(y, X).fit()
    return model, X.columns

def lag_search_with_dummies(st, cols, dummies):
    """含虚拟变量的逐滞后 Wald 检验，返回 {lag: p}。cols=[y, x]。"""
    yname, xname = cols
    ps = {}
    for L in range(1, MAXLAG + 1):
        base = st.drop(columns="date")
        y = base[yname].iloc[L:].reset_index(drop=True)
        ylags = pd.concat([base[yname].shift(k).iloc[L:].reset_index(drop=True) for k in range(1, L + 1)], axis=1)
        xlags = pd.concat([base[xname].shift(k).iloc[L:].reset_index(drop=True) for k in range(1, L + 1)], axis=1)
        ylags.columns = [f"{yname}_L{k}" for k in range(1, L + 1)]
        xlags.columns = [f"{xname}_L{k}" for k in range(1, L + 1)]
        D = dummies.iloc[L:].reset_index(drop=True).drop(columns="date")
        yl, xl, Dd = ylags.reset_index(drop=True), xlags.reset_index(drop=True), D.reset_index(drop=True)
        ok = y.notna() & yl.notna().all(axis=1) & xl.notna().all(axis=1) & Dd.notna().all(axis=1)
        X_u = sm.add_constant(pd.concat([yl, xl, Dd], axis=1)[ok])
        X_r = sm.add_constant(pd.concat([yl, Dd], axis=1)[ok])
        yv = y[ok]
        u = OLS(yv, X_u).fit()
        r = OLS(yv, X_r).fit()
        xcols = list(xlags.columns)
        test = u.wald_test(" = ".join(xcols), scalar=True, use_f=False)
        ps[L] = float(test.pvalue)
    best = min(ps, key=ps.get)
    return best, ps[best], ps

rows_lag, rows_coef = [], []
for name in TOPICS:
    m, st = baseline[name]["st"].copy(), None
    st = baseline[name]["st"]
    dummies = build_dummies(st["date"], name)
    D_all = dummies.drop(columns="date")
    D_all = D_all.loc[:, D_all.std() > 0]           # 剔除样本内恒定的虚拟变量
    dropped = [c for c in dummies.columns if c != "date" and c not in D_all.columns]
    if dropped:
        print(f"  [{name}] 恒定剔除: {dropped} (ICD-11 步进变量早于样本起点)")
    for label, cols in [("media_to_public", ["public", "media"]), ("public_to_media", ["media", "public"])]:
        bl = baseline[name][("m2p" if label == "media_to_public" else "p2m")][0]
        ml, mp, _ = lag_search_with_dummies(st, cols, dummies)
        rows_lag.append({"topic": name, "direction": label, "baseline_lag": int(bl),
                         "mv_optimal_lag": int(ml), "lag_change_days": int(ml - bl),
                         "mv_p": f"{mp:.3e}", "significant_at_q05": mp < 0.05})
    # 公众（Wikipedia）方程的虚拟变量系数：在 P→M 最优滞后处的无约束模型
    yname = "public"
    L = int([r for r in rows_lag if r["topic"] == name and r["direction"] == "public_to_media"][0]["mv_optimal_lag"])
    base = st.drop(columns="date")
    y = base[yname].iloc[L:].reset_index(drop=True)
    parts, names = [], []
    for v in ["public", "media"]:
        lagdf = pd.concat([base[v].shift(k).iloc[L:].reset_index(drop=True) for k in range(1, L + 1)], axis=1)
        lagdf.columns = [f"{v}_L{k}" for k in range(1, L + 1)]
        parts.append(lagdf); names += list(lagdf.columns)
    Dd = dummies.drop(columns="date").iloc[L:].reset_index(drop=True)
    Dd = Dd.loc[:, Dd.std() > 0]
    ok = y.notna() & pd.concat(parts + [Dd], axis=1).notna().all(axis=1)
    X = sm.add_constant(pd.concat(parts + [Dd], axis=1)[ok])
    yv = y[ok]
    # 原始尺度
    u = OLS(yv, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    # 标准化尺度（全部变量 z-score 后重拟合，取虚拟变量系数）
    Xz = sm.add_constant(pd.concat(parts + [Dd], axis=1)[ok].apply(lambda c: (c - c.mean()) / c.std()))
    yz = (yv - yv.mean()) / yv.std()
    uz = OLS(yz, Xz).fit()
    for c in Dd.columns:
        rows_coef.append({"topic": name, "equation": "public",
                          "dummy": c, "model_lag": L,
                          "coef_raw": round(float(u.params[c]), 4), "p_raw": f"{u.pvalues[c]:.4f}",
                          "coef_std": round(float(uz.params[c]), 3), "p_std": f"{uz.pvalues[c]:.4f}"})
lag_df = pd.DataFrame(rows_lag); coef_df = pd.DataFrame(rows_coef)
lag_df.to_csv("multivariate_dummies_lagcheck.csv", index=False)
coef_df.to_csv("multivariate_dummies_coefficients.csv", index=False)
print(lag_df.to_string(index=False))
print("\nCOVID 虚拟变量系数（公众方程, 标准化尺度）:")
print(coef_df[coef_df.dummy.str.startswith("covid")][["topic", "dummy", "coef_std", "p_std"]].to_string(index=False))
print("最大滞后变动(含虚拟变量):", lag_df.lag_change_days.abs().max())


# ---------- C. Welch 相干谱 ----------

print("\n== C. Welch 相干（256 天窗, Hann, fs=1/day）==")
coh_rows = []
for name in TOPICS:
    m = baseline[name]["merged"]
    a = ((m["m"] - m["m"].mean()) / m["m"].std()).values
    b = ((m["w"] - m["w"].mean()) / m["w"].std()).values
    f, C = coherence(a, b, fs=1.0, window="hann", nperseg=256)
    period = 1.0 / f
    for fi, pi, ci in zip(f, period, C):
        coh_rows.append({"topic": name, "period_days": round(pi, 2), "coherence": round(float(ci), 4)})
    band = (period >= 2) & (period <= 60)
    pk, _ = find_peaks(C[band], height=np.median(C[band]))
    periods_band = period[band][pk]; coh_band = C[band][pk]
    top = periods_band[np.argsort(coh_band)[::-1][:3]]
    print(f"  {name:<22s} 2–60 天带内最强周期: {[round(t,1) for t in top]}")
coh = pd.DataFrame(coh_rows)
coh.to_csv("cross_spectral_coherence.csv", index=False)


# ---------- D. TLS（时间领先得分）----------

print("\n== D. TLS（80 峰值事件）==")
tls_rows = []
for name in TOPICS:
    m = baseline[name]["merged"].set_index("date")
    evs = json.load(open(f"{name}_events.json"))
    asym = abs(baseline[name]["m2p"][0] - baseline[name]["p2m"][0])
    for ev in evs:
        d = pd.Timestamp(ev["date"])
        if d not in m.index:
            continue
        i = m.index.get_loc(d)
        if i < 5:
            continue
        g_win = m["mentions"].iloc[i-5:i].mean()   # 前 5 天均值（t-5..t-1）
        w_win = m["wikipedia"].iloc[i-5:i].mean()
        g_ratio = m["mentions"].iloc[i] / g_win if g_win > 0 else np.nan
        w_ratio = m["wikipedia"].iloc[i] / w_win if w_win > 0 else np.nan
        # 对数口径变体
        g_log = np.log1p(m["mentions"]).iloc[i] - np.log1p(m["mentions"]).iloc[i-5:i].mean()
        w_log = np.log1p(m["wikipedia"]).iloc[i] - np.log1p(m["wikipedia"]).iloc[i-5:i].mean()
        tls_rows.append({"topic": name, "date": str(ev["date"]), "peak_type": ev["type"],
                         "g_ratio": round(float(g_ratio), 3), "w_ratio": round(float(w_ratio), 3),
                         "topic_asymmetry_days": asym,
                         "g_logdiff": round(float(g_log), 3), "w_logdiff": round(float(w_log), 3)})
ev_df = pd.DataFrame(tls_rows).dropna()
pooled_sd = np.sqrt((ev_df.g_ratio.std()**2 + ev_df.w_ratio.std()**2) / 2)
ev_df["tls"] = ((ev_df.g_ratio - ev_df.w_ratio) / pooled_sd).round(3)
pooled_sd_log = np.sqrt((ev_df.g_logdiff.std()**2 + ev_df.w_logdiff.std()**2) / 2)
ev_df["tls_log"] = ((ev_df.g_logdiff - ev_df.w_logdiff) / pooled_sd_log).round(3)
ev_df.to_csv("tls_events.csv", index=False)

summ = ev_df.groupby("topic").agg(n=("tls", "size"), tls_mean=("tls", "mean"),
                                  tls_median=("tls", "median"), asymmetry=("topic_asymmetry_days", "first")).reset_index()
summ.to_csv("tls_summary_by_topic.csv", index=False)
rs_raw, p_raw = spearmanr(ev_df["tls"].abs(), ev_df["topic_asymmetry_days"])
rs_log, p_log = spearmanr(ev_df["tls_log"].abs(), ev_df["topic_asymmetry_days"])
print(f"  原始口径: rs={rs_raw:.3f} (P={p_raw:.4f}) | 对数口径: rs={rs_log:.3f} (P={p_log:.4f}) | 正文声称 rs=.34 (P=.002)")
print(summ.to_string(index=False))


# ---------- E. loot box 滚动窗 ----------

print("\n== E. loot box 滚动窗（365 天, 步长 30 天, P→M）==")
m = baseline["loot_box"]["merged"].reset_index(drop=True)
rows = []
for start in range(0, len(m) - 365 + 1, 30):
    win = m.iloc[start:start + 365]
    s = pd.concat([make_stationary(np.log1p(win["mentions"])),
                   make_stationary(np.log1p(win["wikipedia"]))], axis=1).dropna()
    s.columns = ["media", "public"]
    if len(s) < 300:
        rows.append({"window_start": str(win.date.iloc[0].date()), "window_end": str(win.date.iloc[-1].date()),
                     "n_obs": len(s), "p_lag1": "", "p_min_30lags": "", "significant_05": ""})
        continue
    gc1 = grangercausalitytests(s[["media", "public"]], 1)
    p1 = float(gc1[1][0]["ssr_chi2test"][1])
    gc30 = grangercausalitytests(s[["media", "public"]], 30)
    pmin = min(float(gc30[L][0]["ssr_chi2test"][1]) for L in range(1, 31))
    rows.append({"window_start": str(win.date.iloc[0].date()), "window_end": str(win.date.iloc[-1].date()),
                 "n_obs": len(s), "p_lag1": f"{p1:.4g}", "p_min_30lags": f"{pmin:.4g}",
                 "significant_05": pmin < 0.05})
rw = pd.DataFrame(rows)
rw.to_csv("rolling_window_lootbox.csv", index=False)
sig = rw[rw.significant_05 != ""]
share = (sig.significant_05 == True).mean() * 100 if len(sig) else float("nan")
share_lag1 = (sig.p_lag1.astype(float) < 0.05).mean() * 100 if len(sig) else float("nan")
print(f"  有效窗口 {len(sig)} | 逐窗最小 p (1–30 阶) P<.05 占比 {share:.1f}% | 固定 lag-1 P<.05 占比 {share_lag1:.1f}% | 正文声称 78%")
print("\n完成。产出 7 份 CSV。")
