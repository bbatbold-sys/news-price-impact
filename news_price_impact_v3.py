"""
News -> Price Impact ML Pipeline  v3
======================================
Accuracy upgrades over v2:
  1. Lag return features   — what was the asset doing BEFORE the news? (T-1, T-3, T-5)
  2. Rolling sentiment     — 7-day avg FinBERT score per asset (sentiment momentum)
  3. Class rebalancing     — FLAT class is overrepresented; fix with class weights
  4. Per-class models      — separate XGBoost for stocks / crypto / commodities
  5. LightGBM ensemble     — blend XGBoost + LightGBM predictions
  6. Optuna tuning         — find best hyperparameters automatically
  7. Better text features  — earnings keywords, numbers, upgrade/downgrade signals
  8. Proper time-split     — train on 2024-04 to 2025-06, test on 2025-06 to 2026-02
"""

import os, warnings, sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from news_price_impact_v2 import (
    load_market_data, compute_all_technicals, load_news_data,
    score_finbert, match_news_to_assets, add_news_volume,
    compute_price_impacts, build_features,
    MARKET_PATH, NEWS_PATH, HORIZONS, MODEL_DIR,
    CRYPTO_ASSETS, COMMODITY_ASSETS, TECH_COLS,
    _lookup_technicals
)

TRAIN_CUTOFF = pd.Timestamp("2025-06-01")
V3_MODEL_DIR = "models_v3"
os.makedirs(V3_MODEL_DIR, exist_ok=True)

# ── EXTRA TEXT FEATURES ───────────────────────────────────────────────────────
BULLISH_KW  = ["beats","beat","raises","upgrade","upgraded","record","surges",
               "rallies","growth","profit","dividend","buyback","outperform",
               "strong","boom","soars","jumps","milestone","breakthrough"]
BEARISH_KW  = ["misses","missed","cuts","downgrade","downgraded","warning",
               "lawsuit","investigation","recall","bankrupt","layoffs","fired",
               "loss","decline","drops","plunges","fraud","breach","shortage"]
NUMBER_PAT  = r'\d+\.?\d*\s*(%|billion|million|bps|basis)'

def extract_text_features(df):
    df = df.copy()
    tl = df["title"].str.lower().fillna("")
    df["has_bullish_kw"]  = tl.apply(lambda t: int(any(k in t for k in BULLISH_KW)))
    df["has_bearish_kw"]  = tl.apply(lambda t: int(any(k in t for k in BEARISH_KW)))
    df["has_number"]      = tl.str.contains(NUMBER_PAT, regex=True).astype(int)
    df["title_len"]       = df["title"].str.len().fillna(0)
    df["kw_signal"]       = df["has_bullish_kw"] - df["has_bearish_kw"]   # +1/-1/0
    return df

# ── LAG RETURN FEATURES ───────────────────────────────────────────────────────
def add_lag_returns(df, prices):
    """Add price returns in the days BEFORE each news event."""
    print("  Adding lag return features...")
    lag_cols = {"lag_ret1": 1, "lag_ret3": 3, "lag_ret5": 5}
    for col in lag_cols:
        df[col] = np.nan

    for i, (idx, row) in enumerate(df.iterrows()):
        asset = row["asset"]
        date  = row["date"]
        if asset not in prices.columns:
            continue
        past = prices.index[prices.index <= date]
        if len(past) < 6:
            continue
        p_now = prices.loc[past[-1], asset]
        for col, lag in lag_cols.items():
            if len(past) > lag:
                p_lag = prices.loc[past[-(lag+1)], asset]
                if p_lag and p_lag != 0 and not pd.isna(p_lag):
                    df.at[idx, col] = (p_now - p_lag) / p_lag * 100
    return df

# ── ROLLING SENTIMENT ─────────────────────────────────────────────────────────
def add_rolling_sentiment(df):
    """7-day rolling mean FinBERT score per asset (sentiment momentum)."""
    print("  Adding rolling sentiment...")
    df = df.sort_values(["asset", "date"]).copy()
    df["roll_sent7"] = (df.groupby("asset")["finbert_compound"]
                          .transform(lambda x: x.rolling(7, min_periods=1).mean()))
    return df

# ── FULL FEATURE ENGINEERING (v3) ────────────────────────────────────────────
def build_features_v3(df, technicals, prices):
    # Start with v2 features
    df, feature_cols_v2 = build_features(df, technicals)

    # Add v3 extras
    df = extract_text_features(df)
    df = add_lag_returns(df, prices)
    df = add_rolling_sentiment(df)

    extra_cols = ["has_bullish_kw", "has_bearish_kw", "has_number",
                  "title_len", "kw_signal",
                  "lag_ret1", "lag_ret3", "lag_ret5",
                  "roll_sent7"]
    for col in extra_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    feature_cols = feature_cols_v2 + extra_cols
    joblib.dump(feature_cols, f"{V3_MODEL_DIR}/feature_cols_v3.pkl")
    return df, feature_cols

# ── OPTUNA TUNING ─────────────────────────────────────────────────────────────
def tune_xgb(X_tr, y_tr, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": 42, "verbosity": 0,
        }
        m  = xgb.XGBRegressor(**params)
        cv = cross_val_score(m, X_tr, y_tr, cv=3, scoring="r2")
        return cv.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# ── PER-CLASS MODELS ──────────────────────────────────────────────────────────
def get_asset_class(asset):
    return ("crypto" if asset in CRYPTO_ASSETS else
            "commodity" if asset in COMMODITY_ASSETS else "stock")

# ── TRAIN MODELS (v3) ─────────────────────────────────────────────────────────
def train_v3(df, feature_cols, tune=True):
    """
    Train per-class blended (XGBoost + LightGBM) models with class rebalancing.
    Returns nested dict: models[horizon][asset_class] = {reg, clf, lgb_reg, lgb_clf}
    """
    models  = {}
    summary = {}
    classes = ["stock", "crypto", "commodity"]

    for h in HORIZONS:
        target  = f"return_T{h}"
        models[h] = {}

        all_true_cls, all_pred_cls = [], []
        all_true_ret, all_pred_ret = [], []

        for cls in classes:
            cls_mask = df["asset"].apply(get_asset_class) == cls
            sub = df[cls_mask][feature_cols + [target]].dropna(subset=[target])
            if len(sub) < 50:
                continue

            X, y   = sub[feature_cols], sub[target]
            y_cls  = pd.cut(y, bins=[-np.inf,-1,1,np.inf], labels=[0,1,2]).astype(int)

            # Class weights (penalise FLAT less than UP/DOWN)
            class_counts = np.bincount(y_cls)
            weights      = len(y_cls) / (3 * class_counts)
            sample_w     = np.array([weights[c] for c in y_cls])

            # Optuna tune (only on first horizon to save time)
            if tune and h == 1 and len(X) > 200:
                print(f"    Tuning XGB for {cls} T+{h} ({len(X)} samples)...")
                best_p = tune_xgb(X, y, n_trials=25)
            else:
                best_p = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.5, reg_lambda=1.0)

            # XGBoost regression
            xgb_reg = xgb.XGBRegressor(**best_p, random_state=42, verbosity=0)
            xgb_reg.fit(X, y)

            # XGBoost classifier with class weights
            xgb_clf = xgb.XGBClassifier(
                **{k:v for k,v in best_p.items()},
                random_state=42, verbosity=0, eval_metric="mlogloss"
            )
            xgb_clf.fit(X, y_cls, sample_weight=sample_w)

            # LightGBM regression
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0,
                random_state=42, verbose=-1
            )
            lgb_reg.fit(X, y)

            # LightGBM classifier
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42, verbose=-1
            )
            lgb_clf.fit(X, y_cls)

            models[h][cls] = {
                "xgb_reg": xgb_reg, "xgb_clf": xgb_clf,
                "lgb_reg": lgb_reg, "lgb_clf": lgb_clf,
                "feature_cols": feature_cols,
            }
            print(f"  T+{h} {cls}: trained on {len(X):,} samples")

        joblib.dump(models[h], f"{V3_MODEL_DIR}/models_T{h}.pkl")

    joblib.dump(feature_cols, f"{V3_MODEL_DIR}/feature_cols_v3.pkl")
    return models

# ── PREDICT (BLEND) ───────────────────────────────────────────────────────────
def blend_predict(X, asset, models_h, alpha=0.5):
    """Blend XGBoost and LightGBM predictions (alpha for XGB weight)."""
    cls = get_asset_class(asset)
    if cls not in models_h:
        cls = "stock"   # fallback
    m = models_h[cls]

    ret_xgb  = m["xgb_reg"].predict(X)[0]
    ret_lgb  = m["lgb_reg"].predict(X)[0]
    ret_blend = alpha * ret_xgb + (1-alpha) * ret_lgb

    prob_xgb  = m["xgb_clf"].predict_proba(X)[0]
    prob_lgb  = m["lgb_clf"].predict_proba(X)[0]
    prob_blend = alpha * prob_xgb + (1-alpha) * prob_lgb
    cls_pred  = int(np.argmax(prob_blend))

    return ret_blend, cls_pred, prob_blend

# ── EVALUATE ──────────────────────────────────────────────────────────────────
def evaluate_v3(test, models, feature_cols):
    label_map = {0:"DOWN", 1:"FLAT", 2:"UP"}
    summary   = {}
    all_rows  = []

    for h in HORIZONS:
        target = f"return_T{h}"
        sub    = test.dropna(subset=[target]).copy()
        if len(sub) == 0:
            continue

        preds_ret, preds_cls, preds_conf = [], [], []
        for _, row in sub.iterrows():
            X = pd.DataFrame([row[feature_cols]])[feature_cols].fillna(0)
            try:
                ret, cls_p, probs = blend_predict(X, row["asset"], models[h])
            except Exception:
                ret, cls_p, probs = 0, 1, [0.33,0.34,0.33]
            preds_ret.append(ret)
            preds_cls.append(cls_p)
            preds_conf.append(max(probs))

        y_ret = sub[target].values
        y_cls = pd.cut(sub[target], bins=[-np.inf,-1,1,np.inf],
                       labels=[0,1,2]).astype(int).values

        preds_ret = np.array(preds_ret)
        preds_cls = np.array(preds_cls)

        mae  = mean_absolute_error(y_ret, preds_ret)
        r2   = r2_score(y_ret, preds_ret)
        acc  = (preds_cls == y_cls).mean()

        ud_mask = preds_cls != 1
        ud_acc  = (preds_cls[ud_mask] == y_cls[ud_mask]).mean() if ud_mask.any() else np.nan

        summary[f"T+{h}"] = {
            "n_test":        len(sub),
            "MAE_%":         round(mae, 4),
            "R2":            round(r2, 4),
            "Direction_Acc": round(acc, 4),
            "UD_Acc":        round(ud_acc, 4) if not np.isnan(ud_acc) else "-",
            "UD_n":          int(ud_mask.sum()),
        }

        sub["pred_return"] = preds_ret
        sub["pred_class"]  = [label_map[c] for c in preds_cls]
        sub["true_class"]  = [label_map[c] for c in y_cls]
        sub["correct"]     = (preds_cls == y_cls)
        sub["confidence"]  = preds_conf
        sub["horizon"]     = h
        all_rows.append(sub)

        print(f"  T+{h}: MAE={mae:.3f}%  R2={r2:.4f}  Acc={acc:.3f}  UD_Acc={ud_acc:.3f} (n={ud_mask.sum()})")

    return summary, pd.concat(all_rows, ignore_index=True)

# ── COMPARE v2 vs v3 ──────────────────────────────────────────────────────────
def compare_versions(summary_v3):
    v2_results = {
        "T+1":  {"Direction_Acc": 0.4702, "MAE_%": 1.774,  "UD_Acc": 0.3018},
        "T+3":  {"Direction_Acc": 0.3742, "MAE_%": 2.981,  "UD_Acc": 0.3589},
        "T+5":  {"Direction_Acc": 0.5000, "MAE_%": 5.225,  "UD_Acc": 0.4923},
        "T+10": {"Direction_Acc": 0.4235, "MAE_%": 5.313,  "UD_Acc": 0.4242},
    }
    print("\n" + "="*65)
    print(f"{'Horizon':<8} {'v2 Acc':>8} {'v3 Acc':>8} {'D Acc':>8}  "
          f"{'v2 MAE':>8} {'v3 MAE':>8} {'D MAE':>8}")
    print("-"*65)
    for h, m3 in summary_v3.items():
        m2   = v2_results.get(h, {})
        acc2 = m2.get("Direction_Acc", 0)
        acc3 = m3["Direction_Acc"]
        mae2 = m2.get("MAE_%", 0)
        mae3 = m3["MAE_%"]
        d_acc = acc3 - acc2
        d_mae = mae3 - mae2
        flag_acc = "+" if d_acc > 0 else ""
        flag_mae = "+" if d_mae > 0 else ""
        print(f"{h:<8} {acc2:>8.3f} {acc3:>8.3f}  {flag_acc}{d_acc:.3f}  "
              f"{mae2:>7.3f}% {mae3:>7.3f}%  {flag_mae}{d_mae:.3f}%")

# ── CHARTS ────────────────────────────────────────────────────────────────────
def make_charts(all_df, summary):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("v3 Model — XGB + LightGBM Ensemble + Per-class + Lag Features",
                 fontsize=14, fontweight="bold")

    horizons = [f"T+{h}" for h in HORIZONS if f"T+{h}" in summary]
    accs   = [summary[h]["Direction_Acc"] for h in horizons]
    ud_acc = [float(summary[h]["UD_Acc"]) if summary[h]["UD_Acc"] != "-" else 0
              for h in horizons]
    maes   = [summary[h]["MAE_%"]         for h in horizons]
    v2_acc = [0.4702, 0.3742, 0.5000, 0.4235]
    v2_mae = [1.774,  2.981,  5.225,  5.313]

    # Accuracy comparison
    ax = axes[0, 0]
    x  = np.arange(len(horizons))
    ax.bar(x-0.25, v2_acc,  0.23, label="v2 (baseline)", color="#aec7e8")
    ax.bar(x,      accs,    0.23, label="v3 all signals",  color="steelblue")
    ax.bar(x+0.25, ud_acc,  0.23, label="v3 UP/DOWN only", color="darkorange")
    ax.axhline(0.333, color="red",  linestyle="--", linewidth=1, label="Random")
    ax.axhline(0.5,   color="gray", linestyle=":",  linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylim(0, 0.8); ax.set_ylabel("Accuracy")
    ax.set_title("Direction Accuracy: v2 vs v3"); ax.legend(fontsize=7)

    # MAE comparison
    ax = axes[0, 1]
    ax.bar(x-0.2, v2_mae, 0.35, label="v2", color="#aec7e8")
    ax.bar(x+0.2, maes,   0.35, label="v3", color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylabel("MAE %"); ax.set_title("MAE: v2 vs v3"); ax.legend()

    # Confusion matrix T+3
    ax = axes[0, 2]
    s3 = all_df[all_df["horizon"] == 3]
    labels = ["DOWN","FLAT","UP"]
    cm = confusion_matrix(s3["true_class"], s3["pred_class"], labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix T+3 (v3)"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # Accuracy by class
    ax = axes[1, 0]
    s_sub = all_df[all_df["horizon"] == 3]
    cls_acc = s_sub.groupby("asset_class")["correct"].mean()
    cls_acc.plot(kind="bar", ax=ax, color=["darkorange","seagreen","steelblue"], edgecolor="black")
    ax.axhline(0.333, color="red", linestyle="--")
    ax.set_title("v3 Accuracy by Asset Class (T+3)")
    ax.set_ylabel("Accuracy"); ax.tick_params(axis="x", rotation=0)

    # Feature importance (XGB, stocks, T+3)
    ax = axes[1, 1]
    try:
        m = joblib.load(f"{V3_MODEL_DIR}/models_T3.pkl")
        imp  = m["stock"]["xgb_reg"].feature_importances_
        feat = [f.replace("finbert_","FB:").replace("_"," ") for f in feature_cols_global]
        idx  = np.argsort(imp)[-15:]
        colors_fi = ["#ff7f0e" if any(t in feat[i] for t in ["lag","roll","kw","has","len"])
                     else "steelblue" for i in idx]
        ax.barh([feat[i] for i in idx], imp[idx], color=colors_fi)
        ax.set_title("Feature Importance (XGB, stocks, T+3)\norange=new v3 features")
    except Exception:
        ax.text(0.5, 0.5, "N/A", ha="center")

    # P&L simulation
    ax = axes[1, 2]
    s3nf = s3[s3["pred_class"] != "FLAT"].sort_values("date")
    s3nf = s3nf.copy()
    s3nf["trade_ret"] = np.where(s3nf["pred_class"] == "UP",
                                  s3nf[f"return_T3"], -s3nf[f"return_T3"])
    daily = s3nf.groupby("date")["trade_ret"].mean().cumsum()
    ax.plot(daily.index, daily.values, color="steelblue", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Simulated Cumulative P&L (T+3, v3)\nWin rate: "
                 f"{(s3nf['trade_ret']>0).mean():.1%}  "
                 f"Total: {daily.iloc[-1]:+.1f}%")
    ax.set_ylabel("Cumulative % return"); ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = "v3_results_charts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts: {out}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
feature_cols_global = []

def main():
    global feature_cols_global

    print("=" * 60)
    print("NEWS -> PRICE IMPACT  v3  (Accuracy Upgrade)")
    print("=" * 60)

    print("\n[1] Loading market data")
    prices = load_market_data(MARKET_PATH)

    print("\n[2] Technical indicators")
    technicals = compute_all_technicals(prices)

    print("\n[3] Loading news")
    news = load_news_data(NEWS_PATH)

    print("\n[4] FinBERT (cached)")
    news = score_finbert(news)

    print("\n[5] Match to assets")
    news_assets = match_news_to_assets(news)
    news_assets = add_news_volume(news_assets)

    print("\n[6] Price impacts")
    impact_df = compute_price_impacts(news_assets, prices)

    print("\n[7] Feature engineering (v3)")
    impact_df, feature_cols = build_features_v3(impact_df, technicals, prices)
    feature_cols_global = feature_cols
    print(f"  Total features: {len(feature_cols)}")

    # Time-based split
    train = impact_df[impact_df["date"] <  TRAIN_CUTOFF]
    test  = impact_df[impact_df["date"] >= TRAIN_CUTOFF]
    print(f"\nTrain: {len(train):,}  Test: {len(test):,}")

    print("\n[8] Training per-class ensembles")
    models = train_v3(train, feature_cols, tune=True)

    print("\n[9] Out-of-sample evaluation")
    summary, all_df = evaluate_v3(test, models, feature_cols)

    print("\n[10] Charts")
    make_charts(all_df, summary)

    # Save
    all_df.to_csv("v3_predictions.csv", index=False)
    pd.DataFrame(summary).T.to_csv("v3_model_performance.csv")
    print("Saved: v3_predictions.csv, v3_model_performance.csv")

    compare_versions(summary)
    print("\nDone.")

if __name__ == "__main__":
    main()
