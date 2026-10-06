"""
Export BH-FDR q-values for all 300 Granger tests and Ljung-Box residual checks.

Replicates granger_analysis.py's pipeline EXACTLY (log1p -> ADF loop -> concat
-> dropna -> grangercausalitytests, chi-square p at lags 1..30) so that the
optimal lags/p-values match granger_results.csv. Adds:
  1. BH (Benjamini-Hochberg) q-values across all 300 tests
     (30 lags x 5 topics x 2 directions), q=0.05.
  2. Ljung-Box test on the unrestricted model residuals at each topic's
     selected optimal lag (both directions), lags 1..10.

Outputs: granger_fdr_qvalues.csv, ljung_box_results.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings("ignore")

WIKI_DIR, GDELT_DIR = "wiki_esports_data", "gdelt_esports_data"
TOPICS = ["esports", "video_game_addiction", "loot_box", "screen_time", "video_game_violence"]
MAXLAG = 30

def make_stationary(series):
    s = series.copy(); diffs = 0
    while True:
        if adfuller(s.dropna())[1] < 0.05: break
        s = s.diff(); diffs += 1
        if diffs > 3: break
    return s, diffs

rows, lb_rows, hac_rows, optimal = [], [], [], {}
for name in TOPICS:
    g = pd.read_csv(os.path.join(GDELT_DIR, f"{name}.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    w = pd.read_csv(os.path.join(WIKI_DIR, f"{name}.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    merged = pd.merge(g, w, on="date", how="inner")
    merged["m"] = np.log1p(merged["mentions"])
    merged["w"] = np.log1p(merged["views"])
    m_stat, md = make_stationary(merged["m"])
    w_stat, wd = make_stationary(merged["w"])
    data = pd.concat([m_stat, w_stat], axis=1).dropna()
    data.columns = ["media", "public"]
    for label, cols in [("media_to_public", ["public", "media"]), ("public_to_media", ["media", "public"])]:
        gc = grangercausalitytests(data[cols], MAXLAG)
        for lag in range(1, MAXLAG + 1):
            chi2, p = gc[lag][0]["ssr_chi2test"][:2]
            rows.append({"topic": name, "direction": label, "lag": lag,
                         "chi2": round(float(chi2), 4), "p_unadj": float(p)})
        best = min(rows[-MAXLAG:], key=lambda r: r["p_unadj"])
        optimal[(name, label)] = best["lag"]
        p_sel = best["lag"]
        # Ljung-Box on unrestricted-model residuals（实测 gc[lag][1] = [restricted, unrestricted, ndarray]）
        ru = gc[p_sel][1][1]
        resid = np.asarray(ru.resid)
        lb10 = acorr_ljungbox(resid, lags=[10], return_df=True).iloc[0]
        lbopt = acorr_ljungbox(resid, lags=[p_sel], return_df=True).iloc[0]
        c = "lb_p" if "lb_p" in lb10.index else "lb_pvalue"
        lb_rows.append({"topic": name, "direction": label, "optimal_lag": p_sel,
                        "n_obs": int(ru.nobs), "differencing_gdelt": md, "differencing_wiki": wd,
                        "lb10_stat": round(float(lb10["lb_stat"]), 2), "lb10_p": round(float(lb10[c]), 4),
                        "lbopt_stat": round(float(lbopt["lb_stat"]), 2), "lbopt_p": round(float(lbopt[c]), 4),
                        "resid_autocorr_at_5pct": bool(lb10[c] <= 0.05 or lbopt[c] <= 0.05)})
        # HAC（Newey-West）稳健 Wald 检验：X 的 p 阶滞后系数联合为零
        y = data[cols[0]]
        Xdf = pd.DataFrame({"const": 1.0}, index=y.index)
        for L in range(1, p_sel + 1):
            Xdf[f"y{L}"] = y.shift(L)
            Xdf[f"x{L}"] = data[cols[1]].shift(L)
        dxy = pd.concat([y, Xdf], axis=1).dropna()
        import statsmodels.api as sm
        yv = dxy.iloc[:, 0]
        Xv = dxy.iloc[:, 1:]
        fit = sm.OLS(yv, Xv).fit(cov_type="HAC", cov_kwds={"maxlags": p_sel})
        x_idx = [Xv.columns.get_loc(f"x{L}") for L in range(1, p_sel + 1)]
        R = np.zeros((p_sel, Xv.shape[1])); 
        for i, ix in enumerate(x_idx): R[i, ix] = 1
        wt = fit.wald_test(R, scalar=True)
        hac_rows.append({"topic": name, "direction": label, "optimal_lag": p_sel,
                         "chi2_orig": best["chi2"], "p_orig": best["p_unadj"],
                         "wald_hac_chi2": round(float(wt.statistic), 2), "wald_df": p_sel,
                         "p_hac": float(wt.pvalue),
                         "sig_05_orig": best["p_unadj"] < 0.05, "sig_05_hac": bool(wt.pvalue < 0.05)})

df = pd.DataFrame(rows)
# Benjamini-Hochberg across all 300 tests
df = df.sort_values("p_unadj").reset_index(drop=True)
n = len(df)
df["rank"] = np.arange(1, n + 1)
df["bh_threshold"] = df["rank"] / n * 0.05
q = df["p_unadj"] * n / df["rank"]
df["q_BH"] = q[::-1].cummin()[::-1]          # step-up monotonicity
df["sig_at_q05"] = df["p_unadj"] <= df["bh_threshold"]
df = df.sort_values(["topic", "direction", "lag"]).reset_index(drop=True)
cols = ["topic", "direction", "lag", "chi2", "p_unadj", "q_BH", "bh_threshold", "sig_at_q05"]
df[cols].to_csv("granger_fdr_qvalues.csv", index=False)

lb = pd.DataFrame(lb_rows)
lb.to_csv("ljung_box_results.csv", index=False)
hac = pd.DataFrame(hac_rows)
hac.to_csv("hac_robustness_results.csv", index=False)

# ---- sanity checks vs granger_results.csv ----
gr = pd.read_csv("granger_results.csv")
print("== 与 granger_results.csv 对账 ==")
ok = True
for _, r in gr.iterrows():
    for label, lag_col, p_col in [("media_to_public", "media_to_public_lag", "media_to_public_p"),
                                  ("public_to_media", "public_to_media_lag", "public_to_media_p")]:
        row = df[(df.topic == r["topic"]) & (df.direction == label) & (df.lag == r[lag_col])]
        match = len(row) == 1 and abs(row.iloc[0]["p_unadj"] - r[p_col]) < 1e-9
        ok &= match
        print(f"  {r['topic']:<22s} {label:<16s} lag={r[lag_col]:>2d} p={r[p_col]:.6f} match={match}")
print("对账全部一致:", ok)

opt = df[df.apply(lambda r: optimal[(r.topic, r.direction)] == r.lag, axis=1)]
print("\n== 10 个最优滞后的 FDR 状态 ==")
print(opt[["topic", "direction", "lag", "p_unadj", "q_BH", "bh_threshold", "sig_at_q05"]].to_string(index=False))
print("\n300 检验中 q<0.05 的数目:", int(df.sig_at_q05.sum()), "/", n)
print("最大最优滞后 q_BH:", opt.q_BH.max().round(6))
print("\n== Ljung-Box（最优滞后处与 10 阶）==")
print(lb.to_string(index=False))
print("LB 全部 p>0.05:", bool((lb[["lb10_p", "lbopt_p"]] > 0.05).all().all()))
print("\n== HAC（Newey-West）稳健 Wald vs 原始卡方 ==")
print(hac.to_string(index=False))
print("HAC 与原检验结论一致:", bool((hac.sig_05_orig == hac.sig_05_hac).all()))
