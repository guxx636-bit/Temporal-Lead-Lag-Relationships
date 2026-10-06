"""
Threshold backtest for the tiered infodemic monitoring protocol
(Manuscript "Media-Public Lead-Lag ... Infodemic Surveillance", JMIR version).

Protocol under test (from the manuscript, Discussion / Implications section):
  (1) rolling baseline of daily Wikipedia page views
  (2) flag any day where views exceed the rolling mean by >= k standard deviations
  (3) "watch" alert = flagged for 3 consecutive days
Ground truth: 16 annotated peak events per topic (5 topics, *_events.json in the repo).

Grid: k in {1.5, 2.0, 2.5} x match window w in {2, 3, 5} days x
      z-window {30-day (protocol), 90-day (baseline variant)} x
      transform {raw, log1p} x rule {flag, watch}

Metrics per cell (pooled across topics + per topic):
  - detection rate overall and by event type (Wiki peaks = monitored series peaks;
    GDELT peaks = the honest test)
  - annualized false alarm count (flag/alert days not within +/- w days of any event)
  - median lead time in days (event_date - earliest flag date in window; positive = leads)

Run:  python backtest_thresholds.py <repo_dir> <out_dir>
"""
import json, os, sys
import numpy as np
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tll_repo"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tll_backtest"
os.makedirs(OUT, exist_ok=True)

TOPICS = ["loot_box", "screen_time", "video_game_addiction",
          "video_game_violence", "esports"]
THRESHOLDS = [1.5, 2.0, 2.5]
WINDOWS = [2, 3, 5]
Z_WINDOWS = [30, 90]
TRANSFORMS = ["raw", "log1p"]
WATCH_RUN = 3  # consecutive flagged days -> watch alert


def load_topic(topic):
    views = pd.read_csv(os.path.join(REPO, f"{topic}.csv"), parse_dates=["date"])
    events = pd.DataFrame(json.load(open(os.path.join(REPO, f"{topic}_events.json"))))
    events["date"] = pd.to_datetime(events["date"])
    return views.sort_values("date").reset_index(drop=True), events


def flag_days(series, k, zwin):
    """Days where value exceeds trailing zwin-day mean (excluding today) by >= k SD."""
    mu = series.rolling(zwin, min_periods=zwin).mean().shift(1)
    sd = series.rolling(zwin, min_periods=zwin).std(ddof=1).shift(1)
    z = (series - mu) / sd
    return (z >= k).to_numpy(), z


def watch_days(flags):
    """Days where the flag has been true for WATCH_RUN consecutive days."""
    run = np.zeros(len(flags), dtype=int)
    c = 0
    for i, f in enumerate(flags):
        c = c + 1 if f else 0
        run[i] = c
    return run >= WATCH_RUN


def backtest(views, events, k, w, zwin, transform, rule):
    s = views.set_index("date")["views"].asfreq("D")
    if transform == "log1p":
        s = np.log1p(s)
    flags, z = flag_days(s, k, zwin)
    active = watch_days(flags) if rule == "watch" else flags
    idx = s.index

    ev = events.copy()
    det = []
    for _, row in ev.iterrows():
        d = row["date"]
        lo, hi = d - pd.Timedelta(days=w), d + pd.Timedelta(days=w)
        mask = (idx >= lo) & (idx <= hi) & active
        hits = idx[mask]
        if len(hits):
            first = hits[0]
            lead = (d - first).days  # positive = signal precedes event
            det.append(lead)
        else:
            det.append(None)
    ev["lead"] = det
    ev["detected"] = [x is not None for x in det]

    # false alarms: active signal days not within +/- w of ANY ground-truth event
    ev_days = np.zeros(len(idx), dtype=bool)
    for d in ev["date"]:
        lo, hi = d - pd.Timedelta(days=w), d + pd.Timedelta(days=w)
        ev_days |= (idx >= lo) & (idx <= hi)
    fa_days = int((active & ~ev_days).sum())
    span_years = (idx[-1] - idx[0]).days / 365.25
    return ev, fa_days, span_years


rows = []
per_topic_rows = []
for zwin in Z_WINDOWS:
    for transform in TRANSFORMS:
        for rule in ["flag", "watch"]:
            for k in THRESHOLDS:
                pooled_ev = []
                tot_fa, tot_yrs = 0, 0
                for w in WINDOWS:
                    pass  # windows handled below per (w); we loop again inside
                for w in WINDOWS:
                    tot_fa, tot_yrs = 0, 0
                    pooled = []
                    for topic in TOPICS:
                        views, events = load_topic(topic)
                        ev, fa, yrs = backtest(views, events, k, w, zwin, transform, rule)
                        ev["topic"] = topic
                        pooled.append(ev)
                        tot_fa += fa; tot_yrs += yrs
                        gd = ev[ev["type"] == "GDELT峰值"]
                        gd_det = gd[gd["detected"]]
                        pos_leads = [x for x in ev["lead"].dropna() if x == x and x > 0]
                        per_topic_rows.append({
                            "zwin": zwin, "transform": transform, "rule": rule,
                            "k": k, "w": w, "topic": topic,
                            "n_events": len(ev),
                            "det_all": int(ev["detected"].sum()),
                            "det_gdelt": int(ev.loc[ev["type"] == "GDELT峰值", "detected"].sum()),
                            "n_gdelt": int((ev["type"] == "GDELT峰值").sum()),
                            "det_wiki": int(ev.loc[ev["type"] == "Wiki峰值", "detected"].sum()),
                            "n_wiki": int((ev["type"] == "Wiki峰值").sum()),
                            "false_alarm_days": fa, "years": round(yrs, 2),
                            "median_lead": (float(np.median([x for x in ev["lead"].dropna() if x == x]))
                                            if ev["detected"].any() else np.nan),
                            "median_lead_gdelt": (float(np.median(gd_det["lead"].dropna()))
                                                  if len(gd_det) else np.nan),
                            "pct_positive_lead": (round(len(pos_leads) / ev["detected"].sum(), 3)
                                                  if ev["detected"].any() else np.nan),
                            "flag_precision": round(1 - fa / max(1, fa + int(ev["detected"].sum() * 0)), 3),
                        })
                    alld = pd.concat(pooled)
                    leads = [x for x in alld["lead"].dropna().tolist() if x == x]
                    rows.append({
                        "zwin": zwin, "transform": transform, "rule": rule, "k": k, "w": w,
                        "n_events": len(alld),
                        "det_rate": round(alld["detected"].mean(), 3),
                        "det_rate_gdelt": round(alld.loc[alld["type"] == "GDELT峰值", "detected"].mean(), 3),
                        "det_rate_wiki": round(alld.loc[alld["type"] == "Wiki峰值", "detected"].mean(), 3),
                        "false_alarm_per_year": round(tot_fa / tot_yrs, 1),
                        "median_lead_days": (int(np.median(leads)) if leads and not np.isnan(np.median(leads)) else np.nan),
                    })

pooled_df = pd.DataFrame(rows)
topic_df = pd.DataFrame(per_topic_rows)
pooled_df.to_csv(os.path.join(OUT, "backtest_pooled.csv"), index=False)
topic_df.to_csv(os.path.join(OUT, "backtest_by_topic.csv"), index=False)

# ---- headline: protocol variant (30-day z, raw, flag) full grid ----
print("=" * 100)
print("HEADLINE — protocol as written (30-day rolling z, raw views, single-day flag), pooled 5 topics / 80 events")
h = pooled_df[(pooled_df["zwin"] == 30) & (pooled_df["transform"] == "raw") & (pooled_df["rule"] == "flag")]
print(h.to_string(index=False))
print()
print("watch-alert variant (3 consecutive days)")
h2 = pooled_df[(pooled_df["zwin"] == 30) & (pooled_df["transform"] == "raw") & (pooled_df["rule"] == "watch")]
print(h2.to_string(index=False))
print()
print("log1p sensitivity")
h3 = pooled_df[(pooled_df["zwin"] == 30) & (pooled_df["transform"] == "log1p") & (pooled_df["rule"] == "flag")]
print(h3.to_string(index=False))
print()
print("90-day baseline sensitivity (raw, flag)")
h4 = pooled_df[(pooled_df["zwin"] == 90) & (pooled_df["transform"] == "raw") & (pooled_df["rule"] == "flag")]
print(h4.to_string(index=False))

# recommended cells from the manuscript: screen_time @2.5, loot_box @1.5 (w=3)
print()
print("=" * 100)
print("MANUSCRIPT-RECOMMENDED CELLS (w=3, 30-day z, raw, flag):")
for topic, k in [("screen_time", 2.5), ("loot_box", 1.5)]:
    r = topic_df[(topic_df["topic"] == topic) & (topic_df["k"] == k) & (topic_df["w"] == 3) &
                 (topic_df["zwin"] == 30) & (topic_df["transform"] == "raw") & (topic_df["rule"] == "flag")]
    print(r.to_string(index=False))
print()
print("saved:", os.path.join(OUT, "backtest_pooled.csv"), "and backtest_by_topic.csv")
