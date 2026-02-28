"""
Historical Backtest — Out-of-Time Accuracy Test
=================================================
The v3 model was trained on news from 2024-04-02 to 2025-06-01.
This script tests it on news from 2021-03-01 to 2024-04-01 —
a completely different time period the model has NEVER seen.

Steps:
  1. Fetch old Yahoo Finance news from GDELT (2021-03 to 2024-03)
  2. Score with FinBERT (cache results)
  3. Build all features (technicals, lags, rolling sentiment)
  4. Predict using saved v3 models — NO retraining
  5. Compare predicted vs actual price moves
  6. Show accuracy broken down by year, asset, asset class
  7. Compare vs the in-sample and recent-period results
"""

import os, sys, time, warnings, json, requests
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from news_price_impact_v2 import (
    load_market_data, compute_all_technicals,
    score_finbert, match_news_to_assets, add_news_volume,
    compute_price_impacts, build_features,
    MARKET_PATH, HORIZONS,
    CRYPTO_ASSETS, COMMODITY_ASSETS
)
from news_price_impact_v3 import (
    build_features_v3, blend_predict, get_asset_class, extract_text_features,
    add_lag_returns, add_rolling_sentiment, V3_MODEL_DIR
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
OLD_NEWS_START  = datetime(2021, 3,  1)
OLD_NEWS_END    = datetime(2024, 4,  1)
OLD_NEWS_CSV    = "gdelt_yahoo_finance_historical.csv"
OLD_FINBERT_CSV = "finbert_scores_historical.csv"
OUT_DIR         = "historical_backtest_results"
GDELT_API       = "https://api.gdeltproject.org/api/v2/doc/doc"
CHUNK_DAYS      = 7
REQUEST_DELAY   = 2.0

os.makedirs(OUT_DIR, exist_ok=True)

# ── STEP 1: FETCH OLD GDELT NEWS ─────────────────────────────────────────────
def fetch_old_news():
    if os.path.exists(OLD_NEWS_CSV):
        df = pd.read_csv(OLD_NEWS_CSV)
        print(f"Loaded cached historical news: {len(df):,} articles")
        return df

    print(f"Fetching Yahoo Finance news from GDELT: {OLD_NEWS_START.date()} to {OLD_NEWS_END.date()}")
    chunks, cur = [], OLD_NEWS_START
    while cur < OLD_NEWS_END:
        end = min(cur + timedelta(days=CHUNK_DAYS), OLD_NEWS_END)
        chunks.append((cur, end))
        cur = end
    print(f"Total chunks: {len(chunks)}")

    all_articles, seen_urls = [], set()
    for i, (cs, ce) in enumerate(chunks, 1):
        params = {
            "query":         "domain:finance.yahoo.com",
            "mode":          "artlist",
            "maxrecords":    250,
            "startdatetime": cs.strftime("%Y%m%d%H%M%S"),
            "enddatetime":   ce.strftime("%Y%m%d%H%M%S"),
            "format":        "json",
            "sort":          "DateDesc",
        }
        try:
            r = requests.get(GDELT_API, params=params, timeout=30)
            r.raise_for_status()
            arts = r.json().get("articles", [])
            new  = sum(1 for a in arts if a.get("url","") not in seen_urls)
            for a in arts:
                if a.get("url","") not in seen_urls:
                    seen_urls.add(a["url"])
                    all_articles.append(a)
            print(f"[{i}/{len(chunks)}] {cs.date()} to {ce.date()}  +{new} (total {len(all_articles)})")
        except Exception as e:
            print(f"[{i}/{len(chunks)}] {cs.date()}  Error: {e}")
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_articles)
    df.to_csv(OLD_NEWS_CSV, index=False)
    print(f"Saved {len(df):,} articles to {OLD_NEWS_CSV}")
    return df

# ── STEP 2: PARSE OLD NEWS ───────────────────────────────────────────────────
def parse_old_news(raw_df):
    df = raw_df[["title","seendate"]].dropna().copy()
    df["date"]        = pd.to_datetime(df["seendate"].str[:8], format="%Y%m%d", errors="coerce")
    df["title_lower"] = df["title"].str.lower()
    df = df.dropna(subset=["date","title"])
    print(f"Parsed: {len(df):,} articles  {df['date'].min().date()} to {df['date'].max().date()}")
    return df

# ── STEP 3: FINBERT (separate cache for historical) ───────────────────────────
def score_finbert_historical(df):
    """Same as score_finbert but uses a separate cache file."""
    from news_price_impact_v2 import FINBERT_CACHE as CURRENT_CACHE
    import news_price_impact_v2 as v2mod

    orig = v2mod.FINBERT_CACHE
    v2mod.FINBERT_CACHE = OLD_FINBERT_CSV
    df = score_finbert(df)
    v2mod.FINBERT_CACHE = orig
    return df

# ── STEP 4: PREDICT WITH SAVED v3 MODELS ─────────────────────────────────────
def predict_with_v3(test_df, feature_cols, models_per_horizon):
    label_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
    summary   = {}
    all_rows  = []

    for h in HORIZONS:
        target = f"return_T{h}"
        sub    = test_df.dropna(subset=[target]).copy()
        if len(sub) == 0:
            continue

        preds_ret, preds_cls, preds_conf = [], [], []
        for _, row in sub.iterrows():
            X = pd.DataFrame([row[feature_cols]])[feature_cols].fillna(0)
            try:
                ret, cls_p, probs = blend_predict(X, row["asset"], models_per_horizon[h])
            except Exception:
                ret, cls_p, probs = 0, 1, [0.33, 0.34, 0.33]
            preds_ret.append(ret)
            preds_cls.append(cls_p)
            preds_conf.append(max(probs))

        y_ret      = sub[target].values
        preds_ret  = np.array(preds_ret)
        preds_cls  = np.array(preds_cls)
        y_cls      = pd.cut(sub[target], bins=[-np.inf,-1,1,np.inf],
                            labels=[0,1,2]).astype(int).values

        mae = mean_absolute_error(y_ret, preds_ret)
        r2  = r2_score(y_ret, preds_ret)
        acc = (preds_cls == y_cls).mean()
        ud_mask = preds_cls != 1
        ud_acc  = (preds_cls[ud_mask] == y_cls[ud_mask]).mean() if ud_mask.any() else np.nan

        summary[f"T+{h}"] = {
            "n":             len(sub),
            "MAE_%":         round(mae, 4),
            "R2":            round(r2, 4),
            "Direction_Acc": round(acc, 4),
            "UD_Acc":        round(ud_acc, 4) if not np.isnan(ud_acc) else "-",
            "UD_n":          int(ud_mask.sum()),
        }

        sub = sub.copy()
        sub["pred_return"] = preds_ret
        sub["pred_class"]  = [label_map[c] for c in preds_cls]
        sub["true_class"]  = [label_map[c] for c in y_cls]
        sub["correct"]     = (preds_cls == y_cls)
        sub["confidence"]  = preds_conf
        sub["horizon"]     = h
        all_rows.append(sub)

        print(f"  T+{h}: MAE={mae:.3f}%  R2={r2:.4f}  Acc={acc:.3f}  "
              f"UD_Acc={ud_acc:.3f} (n={ud_mask.sum():,})")

    return summary, pd.concat(all_rows, ignore_index=True)

# ── STEP 5: PER-YEAR ACCURACY ─────────────────────────────────────────────────
def per_year_accuracy(all_df):
    all_df = all_df.copy()
    all_df["year"] = pd.to_datetime(all_df["date"]).dt.year
    rows = []
    for (year, h), grp in all_df.groupby(["year","horizon"]):
        acc = grp["correct"].mean()
        n   = len(grp)
        rows.append({"year": year, "horizon": f"T+{h}", "accuracy": acc, "n": n})
    return pd.DataFrame(rows)

# ── STEP 6: CHARTS ────────────────────────────────────────────────────────────
def make_charts(summary_hist, all_df, year_acc):
    # Known results from other periods
    in_sample = {   # from v2 random split (inflated)
        "T+1":  0.581, "T+3": 0.573, "T+5": 0.624, "T+10": 0.622
    }
    recent_oos = {  # v3 backtest on 2025-06 to 2026-02
        "T+1":  0.444, "T+3": 0.393, "T+5": 0.454, "T+10": 0.411
    }
    hist_acc = {h: m["Direction_Acc"] for h, m in summary_hist.items()}

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Historical Backtest — Model Trained 2024-2025, Tested on 2021-2024",
                 fontsize=14, fontweight="bold")

    horizons = [f"T+{h}" for h in HORIZONS if f"T+{h}" in summary_hist]

    # 1. Accuracy across all 3 periods
    ax = axes[0, 0]
    x  = np.arange(len(horizons))
    ax.bar(x - 0.27, [in_sample.get(h,0)   for h in horizons], 0.25,
           label="In-sample (random split, 2024-26)", color="#aec7e8")
    ax.bar(x,        [recent_oos.get(h,0)   for h in horizons], 0.25,
           label="Out-of-sample (2025-26)",            color="steelblue")
    ax.bar(x + 0.27, [hist_acc.get(h,0)     for h in horizons], 0.25,
           label="Historical OOS (2021-24)",            color="darkorange")
    ax.axhline(0.333, color="red",  linestyle="--", linewidth=1, label="Random")
    ax.axhline(0.5,   color="gray", linestyle=":",  linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylim(0, 0.75); ax.set_ylabel("Direction Accuracy")
    ax.set_title("Accuracy Across All Time Periods")
    ax.legend(fontsize=7)
    for i, h in enumerate(horizons):
        ax.text(i+0.27, hist_acc.get(h,0)+0.01, f"{hist_acc.get(h,0):.2f}",
                ha="center", fontsize=7, color="darkorange", fontweight="bold")

    # 2. Accuracy by year
    ax = axes[0, 1]
    pivot = year_acc[year_acc["horizon"] == "T+3"].set_index("year")["accuracy"]
    bars  = ax.bar(pivot.index, pivot.values, color="darkorange", edgecolor="black")
    ax.axhline(0.333, color="red",  linestyle="--", label="Random")
    ax.axhline(0.5,   color="gray", linestyle=":",  label="50%")
    ax.set_title("T+3 Direction Accuracy by Year")
    ax.set_ylabel("Accuracy"); ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, pivot.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.2%}", ha="center", fontsize=9)

    # 3. Confusion matrix (T+3, historical)
    ax = axes[0, 2]
    s3 = all_df[all_df["horizon"] == 3]
    labels = ["DOWN","FLAT","UP"]
    cm = confusion_matrix(s3["true_class"], s3["pred_class"], labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix T+3 — Historical\n(2021-2024 data)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # 4. Accuracy by asset class (historical)
    ax = axes[1, 0]
    s3 = all_df[all_df["horizon"] == 3]
    cls_acc = s3.groupby("asset_class")["correct"].mean()
    cls_acc.plot(kind="bar", ax=ax,
                 color=["darkorange","seagreen","steelblue"], edgecolor="black")
    ax.axhline(0.333, color="red", linestyle="--", label="Random")
    ax.set_title("Historical Accuracy by Asset Class (T+3)")
    ax.set_ylabel("Accuracy"); ax.tick_params(axis="x", rotation=0)
    ax.legend()
    for i, v in enumerate(cls_acc.values):
        ax.text(i, v+0.005, f"{v:.2%}", ha="center", fontsize=9)

    # 5. MAE comparison across periods
    ax = axes[1, 1]
    in_mae  = {"T+1": 1.529, "T+3": 2.891, "T+5": 4.234, "T+10": 4.861}
    oos_mae = {"T+1": 1.760, "T+3": 3.016, "T+5": 4.997, "T+10": 5.681}
    hist_mae = {h: m["MAE_%"] for h, m in summary_hist.items()}
    x = np.arange(len(horizons))
    ax.bar(x-0.27, [in_mae.get(h,0)   for h in horizons], 0.25,
           label="In-sample",     color="#aec7e8")
    ax.bar(x,      [oos_mae.get(h,0)  for h in horizons], 0.25,
           label="Recent OOS",    color="steelblue")
    ax.bar(x+0.27, [hist_mae.get(h,0) for h in horizons], 0.25,
           label="Historical OOS",color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylabel("MAE %"); ax.set_title("MAE Across All Periods")
    ax.legend(fontsize=8)

    # 6. Top assets by historical accuracy
    ax = axes[1, 2]
    s3 = all_df[all_df["horizon"] == 3]
    asset_acc = (s3.groupby("asset")
                   .agg(acc=("correct","mean"), n=("correct","count"))
                   .query("n >= 10")
                   .sort_values("acc", ascending=False)
                   .head(15))
    ax.barh(asset_acc.index, asset_acc["acc"], color="steelblue")
    ax.axvline(0.333, color="red", linestyle="--", label="Random")
    ax.set_title("Top 15 Assets by Historical Accuracy (T+3)")
    ax.set_xlabel("Direction Accuracy"); ax.legend(fontsize=8)
    for i, (v, n) in enumerate(zip(asset_acc["acc"], asset_acc["n"])):
        ax.text(v+0.002, i, f"{v:.0%} (n={n})", va="center", fontsize=7)

    plt.tight_layout()
    out = f"{OUT_DIR}/historical_backtest_charts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts: {out}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HISTORICAL BACKTEST  (2021-03 to 2024-04)")
    print("Model: v3 trained on 2024-04 to 2025-06")
    print("=" * 60)

    # Step 1: fetch / load historical news
    print("\n[1] Fetching historical GDELT news (2021-2024)")
    raw = fetch_old_news()

    print("\n[2] Parsing")
    news = parse_old_news(raw)

    print("\n[3] FinBERT sentiment")
    news = score_finbert_historical(news)

    print("\n[4] Loading market data + technicals")
    prices     = load_market_data(MARKET_PATH)
    technicals = compute_all_technicals(prices)

    print("\n[5] Match to assets")
    news_assets = match_news_to_assets(news)
    news_assets = add_news_volume(news_assets)
    print(f"  {len(news_assets):,} news-asset pairs")

    print("\n[6] Price impacts")
    impact_df = compute_price_impacts(news_assets, prices)

    print("\n[7] Feature engineering (v3)")
    impact_df, feature_cols = build_features_v3(impact_df, technicals, prices)
    print(f"  {len(impact_df):,} rows, {len(feature_cols)} features")

    print("\n[8] Loading saved v3 models")
    models_per_h = {}
    for h in HORIZONS:
        path = f"{V3_MODEL_DIR}/models_T{h}.pkl"
        if os.path.exists(path):
            models_per_h[h] = joblib.load(path)
            print(f"  Loaded T+{h} model")
        else:
            print(f"  WARNING: {path} not found")

    print("\n[9] Predicting on historical data (no retraining)")
    summary, all_df = predict_with_v3(impact_df, feature_cols, models_per_h)

    print("\n[10] Per-year accuracy")
    year_acc = per_year_accuracy(all_df)
    print(year_acc[year_acc["horizon"] == "T+3"].to_string(index=False))

    print("\n[11] Charts")
    make_charts(summary, all_df, year_acc)

    # Save
    all_df.to_csv(f"{OUT_DIR}/predictions.csv", index=False)
    pd.DataFrame(summary).T.to_csv(f"{OUT_DIR}/summary.csv")
    year_acc.to_csv(f"{OUT_DIR}/per_year_accuracy.csv", index=False)
    print(f"Saved to {OUT_DIR}/")

    print("\n" + "=" * 60)
    print("HISTORICAL BACKTEST SUMMARY")
    print("=" * 60)
    print(f"\n{'Horizon':<8} {'n':>7} {'MAE%':>8} {'R2':>8} {'Acc':>8} {'UD_Acc':>8} {'UD_n':>7}")
    print("-" * 60)
    for h, m in summary.items():
        print(f"{h:<8} {m['n']:>7,} {m['MAE_%']:>8.3f} {m['R2']:>8.4f} "
              f"{m['Direction_Acc']:>8.3f} {str(m['UD_Acc']):>8} {m['UD_n']:>7,}")

    print("\nComparison:")
    print(f"  Random baseline:          33.3%")
    print(f"  In-sample (2024-26):      58-62%  (data leakage)")
    print(f"  Recent OOS (2025-26):     39-50%")
    hist_accs = [m["Direction_Acc"] for m in summary.values()]
    print(f"  Historical OOS (2021-24): {min(hist_accs):.1%} - {max(hist_accs):.1%}")

if __name__ == "__main__":
    main()
