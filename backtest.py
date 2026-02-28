"""
Backtest: News -> Price Impact Model  (v2)
==========================================
Proper out-of-sample evaluation using a time-based split:
  - TRAIN  : 2024-04-02  →  2025-06-01  (~14 months)
  - TEST   :  2025-06-01  →  2026-02-27  (~9 months, never seen during training)

Metrics:
  - Direction accuracy (UP / FLAT / DOWN) per horizon
  - MAE & R² for return predictions
  - Confusion matrices
  - Calibration (are confidence scores trustworthy?)
  - Per-asset & per-class breakdown
  - Simulated trading P&L (follow every UP/DOWN signal)
"""

import os, warnings, sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, r2_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# Import everything from v2 so we don't duplicate code
from news_price_impact_v2 import (
    load_market_data, compute_all_technicals, load_news_data,
    score_finbert, match_news_to_assets, add_news_volume,
    compute_price_impacts, build_features,
    MARKET_PATH, NEWS_PATH, HORIZONS, MODEL_DIR,
    CRYPTO_ASSETS, COMMODITY_ASSETS
)

TRAIN_CUTOFF = pd.Timestamp("2025-06-01")   # everything before = train
OUT_DIR      = "backtest_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── BUILD FULL DATASET ────────────────────────────────────────────────────────
def build_dataset():
    print("[1] Loading market data")
    prices = load_market_data(MARKET_PATH)

    print("[2] Technical indicators")
    technicals = compute_all_technicals(prices)

    print("[3] Loading news")
    news = load_news_data(NEWS_PATH)

    print("[4] FinBERT sentiment (cached)")
    news = score_finbert(news)

    print("[5] Matching news to assets")
    news_assets = match_news_to_assets(news)

    print("[6] News volume")
    news_assets = add_news_volume(news_assets)

    print("[7] Price impacts")
    impact_df = compute_price_impacts(news_assets, prices)

    print("[8] Features")
    impact_df, feature_cols = build_features(impact_df, technicals)

    return impact_df, feature_cols, prices

# ── TIME-SPLIT TRAIN / TEST ───────────────────────────────────────────────────
def time_split(df, feature_cols):
    train = df[df["date"] <  TRAIN_CUTOFF].copy()
    test  = df[df["date"] >= TRAIN_CUTOFF].copy()
    print(f"\nTrain: {len(train):,} rows  ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test : {len(test):,}  rows  ({test['date'].min().date()}  to {test['date'].max().date()})")
    return train, test

# ── TRAIN ON TRAIN SPLIT ──────────────────────────────────────────────────────
def train_on_split(train, feature_cols):
    models = {}
    for h in HORIZONS:
        target = f"return_T{h}"
        sub    = train[feature_cols + [target]].dropna(subset=[target])
        X, y   = sub[feature_cols], sub[target]

        reg = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0
        )
        reg.fit(X, y)

        y_cls = pd.cut(y, bins=[-np.inf,-1,1,np.inf], labels=[0,1,2]).astype(int)
        clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, eval_metric="mlogloss"
        )
        clf.fit(X, y_cls)

        models[h] = {"reg": reg, "clf": clf}
        print(f"  Trained T+{h} on {len(sub):,} samples")
    return models

# ── EVALUATE ON TEST SPLIT ────────────────────────────────────────────────────
def evaluate(test, models, feature_cols):
    label_map  = {0: "DOWN", 1: "FLAT", 2: "UP"}
    all_results = []

    summary = {}
    for h in HORIZONS:
        target = f"return_T{h}"
        extra  = [c for c in [target,"date","asset","asset_class","title","finbert_compound"]
                  if c not in feature_cols]
        sub    = test[feature_cols + extra].dropna(subset=[target])
        if len(sub) == 0:
            continue

        X     = sub[feature_cols]
        y_ret = sub[target].values

        reg = models[h]["reg"]
        clf = models[h]["clf"]

        pred_ret   = reg.predict(X)
        pred_cls   = clf.predict(X)
        pred_probs = clf.predict_proba(X)

        y_cls = pd.cut(sub[target], bins=[-np.inf,-1,1,np.inf],
                       labels=[0,1,2]).astype(int).values

        mae    = mean_absolute_error(y_ret, pred_ret)
        r2     = r2_score(y_ret, pred_ret)
        acc    = (pred_cls == y_cls).mean()
        conf   = pred_probs.max(axis=1).mean()

        # Up/Down only (exclude flat predictions) — stronger signal test
        ud_mask  = pred_cls != 1
        ud_acc   = (pred_cls[ud_mask] == y_cls[ud_mask]).mean() if ud_mask.any() else np.nan
        ud_n     = ud_mask.sum()

        summary[f"T+{h}"] = {
            "n_test":          len(sub),
            "MAE_%":           round(mae, 4),
            "R2":              round(r2, 4),
            "Direction_Acc":   round(acc, 4),
            "UD_only_Acc":     round(ud_acc, 4) if not np.isnan(ud_acc) else "-",
            "UD_n_signals":    int(ud_n),
            "Avg_Confidence":  round(conf, 4),
        }

        # Store per-row results
        rows = sub.copy()
        rows[f"pred_return"]   = pred_ret
        rows[f"pred_class"]    = [label_map[c] for c in pred_cls]
        rows[f"true_class"]    = [label_map[c] for c in y_cls]
        rows[f"correct"]       = (pred_cls == y_cls)
        rows[f"confidence"]    = pred_probs.max(axis=1)
        rows[f"prob_up"]       = pred_probs[:, 2]
        rows[f"prob_down"]     = pred_probs[:, 0]
        rows["horizon"]        = h
        all_results.append(rows)

        print(f"  T+{h}: MAE={mae:.3f}%  R2={r2:.4f}  Acc={acc:.3f}  "
              f"UP/DOWN_Acc={ud_acc:.3f} (n={ud_n})")

    all_df = pd.concat(all_results, ignore_index=True)
    return summary, all_df

# ── TRADING SIMULATION ────────────────────────────────────────────────────────
def simulate_trading(all_df, horizon=3):
    """
    Strategy: follow every non-FLAT signal.
      UP   → long  the asset → profit =  actual_return
      DOWN → short the asset → profit = -actual_return
    Compare vs buy-and-hold baseline.
    """
    sub = all_df[all_df["horizon"] == horizon].copy()
    sub = sub[sub["pred_class"] != "FLAT"].copy()
    sub = sub.sort_values("date")

    target_col = f"return_T{horizon}"
    sub["trade_return"] = np.where(
        sub["pred_class"] == "UP",
         sub[target_col],
        -sub[target_col]
    )

    # Daily aggregated P&L (average across signals that day)
    daily = sub.groupby("date").agg(
        n_trades    =("trade_return","count"),
        avg_return  =("trade_return","mean"),
        avg_actual  =(target_col, "mean"),
    ).reset_index()

    daily["cumulative_pnl"]      = daily["avg_return"].cumsum()
    daily["cumulative_baseline"] = daily["avg_actual"].cumsum()

    # Stats
    wins  = (sub["trade_return"] > 0).sum()
    total = len(sub)
    avg   = sub["trade_return"].mean()
    sharpe = sub["trade_return"].mean() / sub["trade_return"].std() * np.sqrt(252) if sub["trade_return"].std() != 0 else 0

    stats = {
        "n_trades":       total,
        "win_rate_%":     round(wins/total*100, 2),
        "avg_trade_%":    round(avg, 4),
        "total_pnl_%":    round(daily["cumulative_pnl"].iloc[-1], 2),
        "sharpe_approx":  round(sharpe, 3),
        "baseline_pnl_%": round(daily["cumulative_baseline"].iloc[-1], 2),
    }
    return daily, stats

# ── CHARTS ────────────────────────────────────────────────────────────────────
def make_backtest_charts(all_df, summary, trading_daily, trading_stats):
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle(f"Backtest Results — Test period: {TRAIN_CUTOFF.date()} to 2026-02-27",
                 fontsize=14, fontweight="bold")

    # 1. Direction accuracy per horizon
    ax = axes[0, 0]
    horizons = [f"T+{h}" for h in HORIZONS if f"T+{h}" in summary]
    accs     = [summary[h]["Direction_Acc"] for h in horizons]
    ud_accs  = [float(summary[h]["UD_only_Acc"]) if summary[h]["UD_only_Acc"] != "-" else 0
                for h in horizons]
    x = np.arange(len(horizons))
    ax.bar(x - 0.2, accs,    0.35, label="All signals", color="steelblue")
    ax.bar(x + 0.2, ud_accs, 0.35, label="UP/DOWN only", color="darkorange")
    ax.axhline(0.333, color="red", linestyle="--", linewidth=1, label="Random (33.3%)")
    ax.axhline(0.5,   color="gray",linestyle=":",  linewidth=1, label="50% line")
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title("Direction Accuracy (Out-of-Sample)")
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(accs, ud_accs)):
        ax.text(i-0.2, a+0.01, f"{a:.1%}", ha="center", fontsize=8)
        ax.text(i+0.2, b+0.01, f"{b:.1%}", ha="center", fontsize=8)

    # 2. Confusion matrix for T+3
    ax = axes[0, 1]
    sub3 = all_df[all_df["horizon"] == 3]
    labels = ["DOWN","FLAT","UP"]
    cm = confusion_matrix(sub3["true_class"], sub3["pred_class"], labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix — T+3 (Out-of-Sample)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # 3. MAE per horizon
    ax = axes[1, 0]
    maes = [summary[h]["MAE_%"] for h in horizons]
    ax.bar(horizons, maes, color="seagreen", edgecolor="black")
    ax.set_title("Mean Absolute Error (%) per Horizon")
    ax.set_ylabel("MAE %")
    for i, v in enumerate(maes):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)

    # 4. Trading P&L simulation
    ax = axes[1, 1]
    ax.plot(trading_daily["date"], trading_daily["cumulative_pnl"],
            label=f"Strategy (T+3)", color="steelblue", linewidth=2)
    ax.plot(trading_daily["date"], trading_daily["cumulative_baseline"],
            label="Baseline (buy&hold)", color="gray", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Simulated Cumulative P&L\n"
                 f"Win rate: {trading_stats['win_rate_%']}%  |  "
                 f"Total: {trading_stats['total_pnl_%']:+.1f}%  |  "
                 f"Sharpe: {trading_stats['sharpe_approx']:.2f}")
    ax.set_ylabel("Cumulative % return"); ax.legend()
    ax.tick_params(axis="x", rotation=30)

    # 5. Accuracy by asset class
    ax = axes[2, 0]
    cls_acc = (all_df[all_df["horizon"].isin([1,3,5])]
               .groupby(["asset_class","horizon"])["correct"].mean()
               .unstack())
    cls_acc.columns = [f"T+{h}" for h in cls_acc.columns]
    cls_acc.plot(kind="bar", ax=ax, edgecolor="black")
    ax.axhline(0.333, color="red", linestyle="--", linewidth=1)
    ax.set_title("Direction Accuracy by Asset Class")
    ax.set_ylabel("Accuracy"); ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1); ax.legend(fontsize=8)

    # 6. Confidence calibration (T+3): are high-confidence predictions more accurate?
    ax = axes[2, 1]
    sub3 = all_df[all_df["horizon"] == 3].copy()
    sub3["conf_bin"] = pd.cut(sub3["confidence"], bins=5)
    calib = sub3.groupby("conf_bin")["correct"].agg(["mean","count"])
    ax.bar(range(len(calib)), calib["mean"], color="darkorange", edgecolor="black")
    ax.set_xticks(range(len(calib)))
    ax.set_xticklabels([str(i) for i in calib.index], rotation=30, fontsize=7)
    ax.axhline(0.333, color="red", linestyle="--", label="Random")
    for i, (acc, n) in enumerate(zip(calib["mean"], calib["count"])):
        ax.text(i, acc + 0.01, f"n={n}", ha="center", fontsize=7)
    ax.set_title("Confidence Calibration (T+3)\nDo higher-confidence predictions win more?")
    ax.set_ylabel("Accuracy"); ax.set_xlabel("Confidence bin"); ax.legend()

    plt.tight_layout()
    out = f"{OUT_DIR}/backtest_charts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts: {out}")

# ── PER-ASSET BREAKDOWN ───────────────────────────────────────────────────────
def per_asset_report(all_df):
    h3 = all_df[all_df["horizon"] == 3]
    report = h3.groupby("asset").agg(
        n_predictions  =("correct","count"),
        accuracy       =("correct","mean"),
        avg_confidence =("confidence","mean"),
        avg_pred_return=("pred_return","mean"),
        avg_true_return=(f"return_T3","mean"),
    ).round(4).sort_values("accuracy", ascending=False)
    report["beat_random"] = report["accuracy"] > 0.333
    return report

# ── SAVE ──────────────────────────────────────────────────────────────────────
def save_backtest(summary, all_df, trading_stats, asset_report):
    pd.DataFrame(summary).T.to_csv(f"{OUT_DIR}/summary_metrics.csv")
    all_df.to_csv(f"{OUT_DIR}/all_predictions.csv", index=False)
    asset_report.to_csv(f"{OUT_DIR}/per_asset_accuracy.csv")
    pd.DataFrame([trading_stats]).to_csv(f"{OUT_DIR}/trading_simulation.csv", index=False)
    print(f"Saved to {OUT_DIR}/")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BACKTEST  —  Time-based out-of-sample evaluation")
    print(f"Train: before {TRAIN_CUTOFF.date()}   Test: after {TRAIN_CUTOFF.date()}")
    print("=" * 60)

    impact_df, feature_cols, prices = build_dataset()

    train, test = time_split(impact_df, feature_cols)

    print("\n[9] Training models on TRAIN split only")
    models = train_on_split(train, feature_cols)

    print("\n[10] Evaluating on TEST split (out-of-sample)")
    summary, all_df = evaluate(test, models, feature_cols)

    print("\n[11] Trading simulation (T+3 signals)")
    trading_daily, trading_stats = simulate_trading(all_df, horizon=3)

    print("\n[12] Per-asset accuracy")
    asset_report = per_asset_report(all_df)
    print(asset_report.head(20).to_string())

    print("\n[13] Charts")
    make_backtest_charts(all_df, summary, trading_daily, trading_stats)

    print("\n[14] Saving")
    save_backtest(summary, all_df, trading_stats, asset_report)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    for h, m in summary.items():
        print(f"\n{h}:  n={m['n_test']}  MAE={m['MAE_%']}%  R2={m['R2']}  "
              f"Acc={m['Direction_Acc']}  UP/DOWN_Acc={m['UD_only_Acc']} (n={m['UD_n_signals']})")

    print("\nTrading Simulation (T+3):")
    for k, v in trading_stats.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
