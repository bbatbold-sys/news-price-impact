"""
News Impact Speed Analysis
===========================
Answers: How fast do prices reflect news?

Method: Event Study (Cumulative Average Return — CAR)
- For each news event, track price at T+0 through T+10 days
- Group by sentiment (positive/negative) and asset class
- Plot CAR curves showing how impact accumulates day by day
- Compute "half-life": how many days until 50% of total impact is absorbed
- Show which assets and classes react fastest
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from news_price_impact_v2 import (
    load_market_data, compute_all_technicals, load_news_data,
    score_finbert, match_news_to_assets, add_news_volume,
    MARKET_PATH, NEWS_PATH,
    CRYPTO_ASSETS, COMMODITY_ASSETS
)

OUT_DIR   = "speed_results"
FINE_DAYS = list(range(1, 11))   # T+1 through T+10 daily
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. COMPUTE FINE-GRAINED RETURNS ──────────────────────────────────────────
def compute_fine_returns(news_assets, prices):
    """Return % change from event date at every day T+1 … T+10."""
    print(f"Computing daily returns T+1..T+10 for {len(news_assets):,} events...")
    records = []
    for i, (_, row) in enumerate(news_assets.iterrows()):
        if i % 5000 == 0:
            print(f"  {i}/{len(news_assets)}")
        asset = row["asset"]
        date  = row["date"]
        if asset not in prices.columns:
            continue
        past = prices.index[prices.index <= date]
        if len(past) == 0:
            continue
        p0 = prices.loc[past[-1], asset]
        if pd.isna(p0) or p0 == 0:
            continue

        future = prices.index[prices.index > date]
        rec = {
            "date":             date,
            "asset":            asset,
            "title":            row["title"],
            "finbert_compound": row["finbert_compound"],
            "is_macro":         row["is_macro"],
            "news_count":       row["news_count"],
            "asset_class":      ("crypto"     if asset in CRYPTO_ASSETS else
                                 "commodity"  if asset in COMMODITY_ASSETS else "stock"),
        }
        for h in FINE_DAYS:
            if len(future) >= h:
                p1 = prices.loc[future[h-1], asset]
                rec[f"ret_T{h}"] = (p1 - p0) / p0 * 100 if not pd.isna(p1) else np.nan
            else:
                rec[f"ret_T{h}"] = np.nan
        records.append(rec)

    df = pd.DataFrame(records)
    # Sentiment bucket
    df["sentiment"] = pd.cut(df["finbert_compound"],
                             bins=[-1.01, -0.1, 0.1, 1.01],
                             labels=["Negative", "Neutral", "Positive"])
    print(f"Done: {len(df):,} rows")
    return df

# ── 2. CAR CURVES ────────────────────────────────────────────────────────────
def car_curves(df):
    """Average return at each day after the event, grouped by sentiment."""
    ret_cols = [f"ret_T{h}" for h in FINE_DAYS]
    result = {}
    for sent in ["Positive", "Negative"]:
        sub = df[df["sentiment"] == sent][ret_cols].mean()
        result[sent] = sub.values
    result["Neutral"] = df[df["sentiment"] == "Neutral"][ret_cols].mean().values
    return result   # {sent: array of len 10}

def car_by_class(df):
    ret_cols = [f"ret_T{h}" for h in FINE_DAYS]
    result = {}
    for cls in ["stock", "crypto", "commodity"]:
        for sent in ["Positive", "Negative"]:
            sub = df[(df["asset_class"] == cls) & (df["sentiment"] == sent)][ret_cols].mean()
            result[f"{cls}_{sent}"] = sub.values
    return result

# ── 3. HALF-LIFE ─────────────────────────────────────────────────────────────
def compute_halflife(car_arr):
    """Day when cumulative return first crosses 50% of final (T+10) value."""
    total = car_arr[-1]
    if total == 0 or np.isnan(total):
        return np.nan
    for i, v in enumerate(car_arr):
        if abs(v) >= abs(total) * 0.5:
            return FINE_DAYS[i]
    return FINE_DAYS[-1]

# ── 4. PER-ASSET SPEED ───────────────────────────────────────────────────────
def per_asset_speed(df):
    ret_cols = [f"ret_T{h}" for h in FINE_DAYS]
    rows = []
    for asset, grp in df.groupby("asset"):
        for sent in ["Positive", "Negative"]:
            sub = grp[grp["sentiment"] == sent][ret_cols]
            if len(sub) < 5:
                continue
            car = sub.mean().values
            hl  = compute_halflife(car)
            rows.append({
                "asset":        asset,
                "sentiment":    sent,
                "n_events":     len(sub),
                "T1_return_%":  round(car[0], 4),
                "T3_return_%":  round(car[2], 4),
                "T5_return_%":  round(car[4], 4),
                "T10_return_%": round(car[-1], 4),
                "halflife_days":hl,
                "pct_impact_day1": round(abs(car[0]) / abs(car[-1]) * 100, 1)
                                   if car[-1] != 0 else np.nan,
            })
    return pd.DataFrame(rows).sort_values("halflife_days")

# ── 5. MACRO VS SPECIFIC ─────────────────────────────────────────────────────
def macro_vs_specific(df):
    ret_cols = [f"ret_T{h}" for h in FINE_DAYS]
    result = {}
    for is_macro, label in [(1, "Macro news"), (0, "Company news")]:
        for sent in ["Positive", "Negative"]:
            sub = df[(df["is_macro"] == is_macro) & (df["sentiment"] == sent)][ret_cols]
            if len(sub) > 0:
                result[f"{label}_{sent}"] = sub.mean().values
    return result

# ── 6. CHARTS ─────────────────────────────────────────────────────────────────
def make_speed_charts(df, car_sent, car_class, car_macro, speed_df):
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("How Fast Do Prices Reflect News?\n(Event Study / CAR Analysis)",
                 fontsize=16, fontweight="bold", y=0.98)

    days = FINE_DAYS

    # ── Chart 1: Overall CAR by sentiment ────────────────────────────────────
    ax1 = fig.add_subplot(4, 2, 1)
    colors = {"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#aec7e8"}
    for sent, arr in car_sent.items():
        ax1.plot(days, arr, marker="o", label=sent, color=colors[sent], linewidth=2)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Avg Cumulative Return After News\n(All assets combined)")
    ax1.set_xlabel("Days after news")
    ax1.set_ylabel("Avg % return")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

    # ── Chart 2: Speed — daily increment (how much moves EACH day) ───────────
    ax2 = fig.add_subplot(4, 2, 2)
    for sent in ["Positive", "Negative"]:
        arr  = car_sent[sent]
        diff = np.diff(arr, prepend=0)   # daily increment
        ax2.bar(np.array(days) + (0.2 if sent == "Positive" else -0.2),
                diff, 0.35, label=sent,
                color=colors[sent], alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Daily Price Increment After News\n(How much moves on each day)")
    ax2.set_xlabel("Days after news"); ax2.set_ylabel("Daily % move")
    ax2.legend(); ax2.grid(alpha=0.3)

    # ── Chart 3: CAR by asset class — Positive news ──────────────────────────
    ax3 = fig.add_subplot(4, 2, 3)
    cls_colors = {"stock": "steelblue", "crypto": "darkorange", "commodity": "seagreen"}
    for key, arr in car_class.items():
        cls, sent = key.rsplit("_", 1)
        if sent == "Positive":
            ax3.plot(days, arr, marker="o", label=cls.capitalize(),
                     color=cls_colors[cls], linewidth=2)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("CAR by Asset Class — POSITIVE news")
    ax3.set_xlabel("Days after news"); ax3.set_ylabel("Avg % return")
    ax3.legend(); ax3.grid(alpha=0.3)

    # ── Chart 4: CAR by asset class — Negative news ──────────────────────────
    ax4 = fig.add_subplot(4, 2, 4)
    for key, arr in car_class.items():
        cls, sent = key.rsplit("_", 1)
        if sent == "Negative":
            ax4.plot(days, arr, marker="o", label=cls.capitalize(),
                     color=cls_colors[cls], linewidth=2, linestyle="--")
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("CAR by Asset Class — NEGATIVE news")
    ax4.set_xlabel("Days after news"); ax4.set_ylabel("Avg % return")
    ax4.legend(); ax4.grid(alpha=0.3)

    # ── Chart 5: % of impact absorbed by Day 1 (speed score) ─────────────────
    ax5 = fig.add_subplot(4, 2, 5)
    spd_pos = speed_df[speed_df["sentiment"] == "Positive"].nlargest(15, "pct_impact_day1")
    ax5.barh(spd_pos["asset"], spd_pos["pct_impact_day1"], color="steelblue")
    ax5.axvline(100, color="red", linestyle="--", linewidth=1)
    ax5.set_title("% of Total Impact Priced In by Day 1\n(Positive news, top 15 fastest)")
    ax5.set_xlabel("% of T+10 impact absorbed on Day 1")
    ax5.grid(alpha=0.3)

    # ── Chart 6: Half-life distribution ──────────────────────────────────────
    ax6 = fig.add_subplot(4, 2, 6)
    hl = speed_df["halflife_days"].dropna()
    ax6.hist(hl, bins=range(1, 12), color="darkorange", edgecolor="black", align="left")
    ax6.set_title("Distribution of Impact Half-Life\n(Days until 50% of total impact)")
    ax6.set_xlabel("Days"); ax6.set_ylabel("Count")
    ax6.set_xticks(FINE_DAYS); ax6.grid(alpha=0.3)

    # ── Chart 7: Macro vs Company news ───────────────────────────────────────
    ax7 = fig.add_subplot(4, 2, 7)
    line_styles = {"Macro news_Positive": ("#2ca02c", "-",  "Macro Positive"),
                   "Macro news_Negative": ("#d62728", "-",  "Macro Negative"),
                   "Company news_Positive": ("#2ca02c", "--", "Company Positive"),
                   "Company news_Negative": ("#d62728", "--", "Company Negative")}
    for key, arr in car_macro.items():
        c, ls, label = line_styles.get(key, ("gray", "-", key))
        ax7.plot(days, arr, color=c, linestyle=ls, label=label, linewidth=2)
    ax7.axhline(0, color="black", linewidth=0.8)
    ax7.set_title("Macro News vs Company-Specific News\n(Speed & Magnitude comparison)")
    ax7.set_xlabel("Days after news"); ax7.set_ylabel("Avg % return")
    ax7.legend(fontsize=7); ax7.grid(alpha=0.3)

    # ── Chart 8: Heatmap — Asset x Day avg return (Positive news) ────────────
    ax8 = fig.add_subplot(4, 2, 8)
    top_assets = (df[df["sentiment"] == "Positive"]
                  .groupby("asset")["ret_T1"].count()
                  .nlargest(15).index)
    ret_cols = [f"ret_T{h}" for h in [1,2,3,5,7,10]]
    hm_data = (df[(df["sentiment"] == "Positive") & (df["asset"].isin(top_assets))]
               .groupby("asset")[ret_cols].mean())
    hm_data.columns = [f"T+{c.split('T')[1]}" for c in ret_cols]
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax8, cbar_kws={"label": "Avg % return"})
    ax8.set_title("Avg Return by Asset & Day (Positive news)\nTop 15 most-covered assets")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{OUT_DIR}/impact_speed_charts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts: {out}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NEWS IMPACT SPEED ANALYSIS")
    print("=" * 60)

    print("\n[1] Loading data")
    prices = load_market_data(MARKET_PATH)
    news   = load_news_data(NEWS_PATH)

    print("\n[2] FinBERT sentiment (cached)")
    news = score_finbert(news)

    print("\n[3] Matching to assets")
    news_assets = match_news_to_assets(news)
    news_assets = add_news_volume(news_assets)

    print("\n[4] Fine-grained daily returns")
    fine_df = compute_fine_returns(news_assets, prices)
    fine_df.to_csv(f"{OUT_DIR}/fine_returns.csv", index=False)

    print("\n[5] CAR curves")
    car_sent  = car_curves(fine_df)
    car_class = car_by_class(fine_df)
    car_macro = macro_vs_specific(fine_df)

    print("\n[6] Per-asset speed")
    speed_df = per_asset_speed(fine_df)
    speed_df.to_csv(f"{OUT_DIR}/asset_speed.csv", index=False)

    print("\n[7] Charts")
    make_speed_charts(fine_df, car_sent, car_class, car_macro, speed_df)

    # Summary stats
    print("\n" + "=" * 60)
    print("SPEED SUMMARY")
    print("=" * 60)
    for sent in ["Positive", "Negative"]:
        arr = car_sent[sent]
        hl  = compute_halflife(arr)
        d1  = abs(arr[0]) / abs(arr[-1]) * 100 if arr[-1] != 0 else 0
        print(f"\n{sent} news:")
        print(f"  T+1 impact:  {arr[0]:+.3f}%")
        print(f"  T+3 impact:  {arr[2]:+.3f}%")
        print(f"  T+10 impact: {arr[-1]:+.3f}%")
        print(f"  % absorbed Day 1: {d1:.1f}%")
        print(f"  Half-life: ~{hl} day(s)")

    print("\nFastest-reacting assets (positive news):")
    fast = speed_df[speed_df["sentiment"] == "Positive"].head(10)
    print(fast[["asset","halflife_days","pct_impact_day1","T1_return_%","T10_return_%"]].to_string(index=False))

    print(f"\nSaved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
