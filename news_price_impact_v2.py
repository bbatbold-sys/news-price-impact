"""
News -> Price Impact ML Pipeline  v2
=====================================
Upgrades over v1:
  - FinBERT financial sentiment  (replaces VADER)
  - Technical indicators: RSI-14, MACD, Bollinger Band position,
    5-day & 20-day momentum, 20-day volatility, SMA crossover
  - Market regime: SPY 20-day return & volatility as context
  - News volume burst: # articles per asset per day
  - Cross-validated XGBoost with early stopping + regularisation
  - Saved models (models/) for re-use without retraining
  - predict_headline() for real-time predictions on new headlines
"""

import os, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MARKET_PATH    = r"C:\Users\batbo\OneDrive\Desktop\Market_Data_5Years.xlsx"
NEWS_PATH      = r"C:\Users\batbo\gdelt_yahoo_finance.csv"
MODEL_DIR      = "models"
FINBERT_CACHE  = "finbert_scores.csv"
HORIZONS       = [1, 3, 5, 10]
os.makedirs(MODEL_DIR, exist_ok=True)

# ── ASSET KEYWORD MAP ─────────────────────────────────────────────────────────
ASSET_KEYWORDS = {
    "NVDA":   ["nvidia","nvda","jensen huang","geforce","cuda"],
    "TSLA":   ["tesla","tsla","elon musk","electric vehicle","cybertruck"],
    "AAPL":   ["apple","aapl","iphone","ipad","tim cook","app store","macbook"],
    "AMD":    ["amd","advanced micro devices","ryzen","radeon"],
    "AMZN":   ["amazon","amzn","aws","jeff bezos","prime"],
    "META":   ["meta","facebook","instagram","zuckerberg","whatsapp","threads"],
    "MSFT":   ["microsoft","msft","windows","azure","copilot","satya nadella","openai"],
    "GOOGL":  ["google","googl","alphabet","youtube","gemini","sundar pichai"],
    "PLTR":   ["palantir","pltr"],
    "INTC":   ["intel","intc"],
    "BABA":   ["alibaba","baba","jack ma","ant group"],
    "SOFI":   ["sofi","social finance"],
    "RIVN":   ["rivian","rivn"],
    "LCID":   ["lucid","lcid"],
    "SNAP":   ["snapchat","snap inc"],
    "SHOP":   ["shopify"],
    "UBER":   ["uber"],
    "NET":    ["cloudflare"],
    "COIN":   ["coinbase"],
    "BAC":    ["bank of america","bac"],
    "JPM":    ["jpmorgan","jp morgan","jamie dimon"],
    "C":      ["citigroup","citibank","citi"],
    "WFC":    ["wells fargo"],
    "MS":     ["morgan stanley"],
    "GS":     ["goldman sachs"],
    "V":      ["visa"],
    "MA":     ["mastercard"],
    "HOOD":   ["robinhood"],
    "SCHW":   ["charles schwab","schwab"],
    "SPY":    ["s&p 500","sp500","spy etf"],
    "QQQ":    ["nasdaq","qqq"],
    "IWM":    ["russell 2000","small cap"],
    "PFE":    ["pfizer"],
    "MRNA":   ["moderna","mrna"],
    "JNJ":    ["johnson & johnson","j&j"],
    "ABBV":   ["abbvie"],
    "LLY":    ["eli lilly","lilly"],
    "XOM":    ["exxon","exxonmobil"],
    "CVX":    ["chevron"],
    "OXY":    ["occidental"],
    "GE":     ["ge aerospace","general electric"],
    "BA":     ["boeing"],
    "DIS":    ["disney"],
    "NFLX":   ["netflix","nflx"],
    "NIO":    ["nio"],
    "F":      ["ford motor"],
    "GME":    ["gamestop","gme"],
    "CL=F":   ["crude oil","wti","oil price","petroleum","opec"],
    "BZ=F":   ["brent crude","brent oil"],
    "NG=F":   ["natural gas","lng"],
    "GC=F":   ["gold","gold price","bullion"],
    "SI=F":   ["silver","silver price"],
    "ZC=F":   ["corn","grain"],
    "ZS=F":   ["soybean","soy"],
    "ZW=F":   ["wheat"],
    "HG=F":   ["copper"],
    "KC=F":   ["coffee","arabica"],
    "BTC-USD":["bitcoin","btc","crypto","cryptocurrency"],
    "ETH-USD":["ethereum","eth","ether"],
    "USDT-USD":["tether","usdt","stablecoin"],
    "SOL-USD":["solana","sol"],
    "XRP-USD":["xrp","ripple"],
    "_MACRO": ["fed","federal reserve","interest rate","inflation","recession",
               "gdp","unemployment","cpi","tariff","trade war","debt ceiling",
               "rate hike","rate cut","jerome powell","treasury","fiscal"],
}
MACRO_ASSETS = ["NVDA","TSLA","AAPL","AMD","AMZN","META","MSFT","GOOGL",
                "SPY","QQQ","BTC-USD","ETH-USD","GC=F","CL=F"]
CRYPTO_ASSETS     = {"BTC-USD","ETH-USD","USDT-USD","SOL-USD","XRP-USD"}
COMMODITY_ASSETS  = {"CL=F","BZ=F","NG=F","GC=F","SI=F","ZC=F","ZS=F","ZW=F","HG=F","KC=F"}

TECH_COLS = ["rsi14","macd_hist","bb_pos","mom5","mom20","vol20","sma_cross","spy_ret20","spy_vol20"]

# ── 1. LOAD MARKET DATA ───────────────────────────────────────────────────────
def load_market_data(path):
    xl = pd.ExcelFile(path)
    frames = {}
    for sheet in ["Stocks","Commodities","Crypto"]:
        df = xl.parse(sheet, header=None)
        df.columns = df.iloc[1]
        df = df.iloc[2:].reset_index(drop=True)
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        new_cols = {"Date": "Date"}
        for c in df.columns[1:]:
            if pd.notna(c) and "\n" in str(c):
                new_cols[c] = str(c).split("\n")[0].strip()
            elif pd.notna(c):
                new_cols[c] = str(c).strip()
        df = df.rename(columns=new_cols)
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("Date").sort_index()
        frames[sheet] = df
    all_prices = pd.concat([frames["Stocks"],frames["Commodities"],frames["Crypto"]], axis=1)
    all_prices = all_prices.loc[~all_prices.index.duplicated(keep="first")]
    print(f"Market data: {len(all_prices)} trading days, {len(all_prices.columns)} assets")
    return all_prices

# ── 2. TECHNICAL INDICATORS ───────────────────────────────────────────────────
def _rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd_hist(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def _bb_pos(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    bw  = (2 * std).replace(0, np.nan)
    return ((series - (sma - std)) / bw).clip(0, 1)

def compute_all_technicals(prices):
    print("Computing technical indicators...")
    technicals = {}
    spy_ret20 = spy_vol20 = None
    if "SPY" in prices.columns:
        spy = prices["SPY"].dropna()
        spy_ret20 = spy.pct_change(20) * 100
        spy_vol20 = spy.pct_change().rolling(20).std() * 100

    for asset in prices.columns:
        series = prices[asset].dropna()
        if len(series) < 60:
            continue
        df = pd.DataFrame(index=series.index)
        df["rsi14"]     = _rsi(series)
        df["macd_hist"] = _macd_hist(series)
        df["bb_pos"]    = _bb_pos(series)
        df["mom5"]      = series.pct_change(5)  * 100
        df["mom20"]     = series.pct_change(20) * 100
        df["vol20"]     = series.pct_change().rolling(20).std() * 100
        df["sma_cross"] = (series.rolling(20).mean() > series.rolling(50).mean()).astype(float)
        if spy_ret20 is not None:
            df["spy_ret20"] = spy_ret20.reindex(df.index)
            df["spy_vol20"] = spy_vol20.reindex(df.index)
        technicals[asset] = df

    print(f"  Done for {len(technicals)} assets")
    return technicals

def _lookup_technicals(technicals, asset, date):
    if asset not in technicals:
        return {}
    df   = technicals[asset]
    past = df.index[df.index <= date]
    if len(past) == 0:
        return {}
    return df.loc[past[-1]].to_dict()

# ── 3. LOAD NEWS ──────────────────────────────────────────────────────────────
def load_news_data(path):
    df = pd.read_csv(path)[["title","seendate"]].dropna()
    df["date"] = pd.to_datetime(df["seendate"].str[:8], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date","title"])
    df["title_lower"] = df["title"].str.lower()
    print(f"News: {len(df)} articles, {df['date'].min().date()} to {df['date'].max().date()}")
    return df

# ── 4. FINBERT SENTIMENT ──────────────────────────────────────────────────────
def score_finbert(df):
    titles = df["title"].tolist()

    # Use cache if available
    if os.path.exists(FINBERT_CACHE):
        cached = pd.read_csv(FINBERT_CACHE)
        cached = cached.drop_duplicates("title")
        merged = df.merge(cached, on="title", how="left")
        missing_mask = merged["finbert_pos"].isna()
        if not missing_mask.any():
            print(f"  FinBERT: loaded all {len(merged)} scores from cache")
            return merged
        titles = merged.loc[missing_mask, "title"].tolist()
        print(f"  FinBERT: {len(titles)} new titles to score")
    else:
        merged = df.copy()
        print(f"  FinBERT: scoring {len(titles)} titles (first run ~10-15 min on CPU)")

    from transformers import pipeline
    pipe = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=-1,
        batch_size=64,
        truncation=True,
        max_length=128,
        top_k=None,
    )

    rows = []
    chunk = 512
    for i in range(0, len(titles), chunk):
        batch = titles[i : i + chunk]
        out   = pipe(batch)
        for title, res in zip(batch, out):
            s = {r["label"]: r["score"] for r in res}
            rows.append({
                "title":           title,
                "finbert_pos":     s.get("positive", 0),
                "finbert_neg":     s.get("negative", 0),
                "finbert_neu":     s.get("neutral",  0),
                "finbert_compound": s.get("positive",0) - s.get("negative",0),
            })
        if i % 2000 == 0:
            print(f"    {i}/{len(titles)}")

    new_scores = pd.DataFrame(rows)

    # Merge with existing cache
    if os.path.exists(FINBERT_CACHE):
        old = pd.read_csv(FINBERT_CACHE)
        new_scores = pd.concat([old, new_scores]).drop_duplicates("title")
    new_scores.to_csv(FINBERT_CACHE, index=False)

    merged = df.merge(
        new_scores[["title","finbert_pos","finbert_neg","finbert_neu","finbert_compound"]],
        on="title", how="left"
    )
    # Fallback VADER for any still-missing
    still_missing = merged["finbert_pos"].isna()
    if still_missing.any():
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        for idx in merged.index[still_missing]:
            c = sia.polarity_scores(str(merged.at[idx,"title"]))["compound"]
            merged.at[idx,"finbert_pos"]      = max(c, 0)
            merged.at[idx,"finbert_neg"]      = max(-c, 0)
            merged.at[idx,"finbert_neu"]      = 1 - abs(c)
            merged.at[idx,"finbert_compound"] = c
    return merged

# ── 5. MATCH NEWS TO ASSETS ───────────────────────────────────────────────────
def match_news_to_assets(df):
    rows = []
    for _, row in df.iterrows():
        title_l  = row["title_lower"]
        matched  = []
        is_macro = any(kw in title_l for kw in ASSET_KEYWORDS["_MACRO"])
        for ticker, kws in ASSET_KEYWORDS.items():
            if ticker == "_MACRO":
                continue
            if any(kw in title_l for kw in kws):
                matched.append(ticker)
        if is_macro and not matched:
            matched = MACRO_ASSETS
        elif is_macro:
            matched = list(set(matched + MACRO_ASSETS))
        if not matched:
            continue
        for asset in matched:
            rows.append({**row.to_dict(), "asset": asset, "is_macro": is_macro})
    result = pd.DataFrame(rows)
    print(f"News-asset pairs: {len(result):,} ({result['asset'].nunique()} assets)")
    return result

# ── 6. NEWS VOLUME ────────────────────────────────────────────────────────────
def add_news_volume(df):
    vol = df.groupby(["asset","date"]).size().reset_index(name="news_count")
    return df.merge(vol, on=["asset","date"], how="left")

# ── 7. PRICE IMPACT ───────────────────────────────────────────────────────────
def _future_return(prices, asset, event_date, h):
    if asset not in prices.columns:
        return np.nan
    future = prices.index[prices.index > event_date]
    past   = prices.index[prices.index <= event_date]
    if len(future) < h or len(past) == 0:
        return np.nan
    p0 = prices.loc[past[-1],    asset]
    p1 = prices.loc[future[h-1], asset]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return (p1 - p0) / p0 * 100

def compute_price_impacts(news_assets, prices):
    print(f"Computing price impacts for {len(news_assets):,} pairs...")
    records = []
    for i, (_, row) in enumerate(news_assets.iterrows()):
        if i % 5000 == 0:
            print(f"  {i}/{len(news_assets)}")
        rec = {k: row[k] for k in
               ["date","title","asset","is_macro","news_count",
                "finbert_pos","finbert_neg","finbert_neu","finbert_compound"]}
        for h in HORIZONS:
            rec[f"return_T{h}"] = _future_return(prices, row["asset"], row["date"], h)
        # Impact speed: first horizon where |return| > 1%
        rec["impact_speed"] = next(
            (h for h in HORIZONS if pd.notna(rec[f"return_T{h}"]) and abs(rec[f"return_T{h}"]) > 1),
            np.nan
        )
        rec["max_abs_return"] = max(
            abs(rec.get(f"return_T{h}") or 0) for h in HORIZONS
        )
        records.append(rec)
    df = pd.DataFrame(records).dropna(subset=["return_T1"])
    print(f"Dataset: {len(df):,} rows")
    return df

# ── 8. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df, technicals):
    df = df.copy()
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["is_macro"]     = df["is_macro"].astype(int)
    df["asset_class"]  = df["asset"].apply(
        lambda a: "crypto" if a in CRYPTO_ASSETS else
                  ("commodity" if a in COMMODITY_ASSETS else "stock")
    )

    le_asset = LabelEncoder().fit(df["asset"])
    le_class = LabelEncoder().fit(df["asset_class"])
    df["asset_enc"]       = le_asset.transform(df["asset"])
    df["asset_class_enc"] = le_class.transform(df["asset_class"])

    # Technical indicators per row
    print("  Attaching technical indicators to each event...")
    tech_rows = [_lookup_technicals(technicals, r["asset"], r["date"])
                 for _, r in df.iterrows()]
    tech_df = pd.DataFrame(tech_rows, index=df.index)
    for col in TECH_COLS:
        df[col] = tech_df[col] if col in tech_df.columns else np.nan
        df[col] = df[col].fillna(df[col].median())

    feature_cols = (
        ["finbert_pos","finbert_neg","finbert_neu","finbert_compound",
         "day_of_week","month","is_macro","news_count",
         "asset_enc","asset_class_enc"]
        + TECH_COLS
    )

    # Save encoders
    joblib.dump(le_asset, f"{MODEL_DIR}/le_asset.pkl")
    joblib.dump(le_class, f"{MODEL_DIR}/le_class.pkl")

    return df, feature_cols

# ── 9. TRAIN MODELS ───────────────────────────────────────────────────────────
def train_models(df, feature_cols):
    summary    = {}
    chart_data = {}

    for h in HORIZONS:
        target = f"return_T{h}"
        sub    = df[feature_cols + [target]].dropna(subset=[target])
        X, y   = sub[feature_cols], sub[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # Regression
        reg = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=30, random_state=42, verbosity=0
        )
        reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred = reg.predict(X_te)
        mae    = mean_absolute_error(y_te, y_pred)
        r2     = r2_score(y_te, y_pred)
        cv_r2  = cross_val_score(
            xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, verbosity=0),
            X, y, cv=5, scoring="r2"
        )

        # Classifier  (0=down, 1=flat, 2=up)
        y_cls = pd.cut(y, bins=[-np.inf,-1,1,np.inf], labels=[0,1,2]).astype(int)
        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)
        clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, random_state=42,
            verbosity=0, eval_metric="mlogloss"
        )
        clf.fit(X_tr_c, y_tr_c, eval_set=[(X_te_c, y_te_c)], verbose=False)
        acc = (clf.predict(X_te_c) == y_te_c).mean()

        joblib.dump(reg, f"{MODEL_DIR}/reg_T{h}.pkl")
        joblib.dump(clf, f"{MODEL_DIR}/clf_T{h}.pkl")

        summary[f"T+{h}"] = {
            "MAE_%":         round(mae, 4),
            "R2":            round(r2, 4),
            "CV_R2_mean":    round(cv_r2.mean(), 4),
            "CV_R2_std":     round(cv_r2.std(), 4),
            "Clf_Accuracy":  round(acc, 4),
            "n_samples":     len(sub),
        }
        chart_data[f"T+{h}"] = {
            "importances": reg.feature_importances_,
            "features":    feature_cols,
            "y_te": y_te, "y_pred": y_pred,
        }
        print(f"  T+{h}: MAE={mae:.3f}%  R2={r2:.4f}  "
              f"CV={cv_r2.mean():.4f}+-{cv_r2.std():.4f}  Acc={acc:.3f}")

    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
    return summary, chart_data

# ── 10. CHARTS ────────────────────────────────────────────────────────────────
def make_charts(df, chart_data, feature_cols):
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle("News -> Price Impact  (v2: FinBERT + Technical Indicators)",
                 fontsize=15, fontweight="bold")

    # Sentiment distribution
    ax = axes[0,0]
    sent = pd.cut(df["finbert_compound"], bins=[-1.01,-0.1,0.1,1.01],
                  labels=["Negative","Neutral","Positive"]).value_counts()
    sent.plot(kind="bar", ax=ax, color=["#d62728","#aec7e8","#2ca02c"], edgecolor="black")
    ax.set_title("FinBERT Sentiment Distribution"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    # Avg return by sentiment
    ax = axes[0,1]
    df["sent_label"] = pd.cut(df["finbert_compound"], bins=[-1.01,-0.1,0.1,1.01],
                               labels=["Negative","Neutral","Positive"])
    si = df.groupby("sent_label")[["return_T1","return_T3","return_T5"]].mean()
    si.plot(kind="bar", ax=ax, color=["steelblue","darkorange","seagreen"])
    ax.set_title("Avg % Return by FinBERT Sentiment")
    ax.set_ylabel("% Return"); ax.axhline(0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["T+1","T+3","T+5"])

    # Feature importance (T+1)
    ax = axes[1,0]
    imp  = chart_data["T+1"]["importances"]
    feat = [f.replace("finbert_","FB:").replace("_enc","").replace("_"," ")
            for f in feature_cols]
    idx  = np.argsort(imp)
    colors = ["steelblue" if "FB:" in feat[i] else
              "#ff7f0e"   if feat[i] in ["rsi14","macd hist","bb pos","mom5",
                                          "mom20","vol20","sma cross"] else
              "#2ca02c"   for i in idx]
    ax.barh([feat[i] for i in idx], imp[idx], color=colors)
    ax.set_title("Feature Importance T+1 (blue=sentiment, orange=technical)")
    ax.set_xlabel("Importance")

    # Impact speed
    ax = axes[1,1]
    spd = df["impact_speed"].dropna()
    ax.hist(spd, bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5],
            color="darkorange", edgecolor="black")
    ax.set_title("Impact Speed: Days Until |Return| > 1%")
    ax.set_xlabel("Days after news"); ax.set_ylabel("Count")

    # Return by asset class
    ax = axes[2,0]
    ci = df.groupby("asset_class")[["return_T1","return_T3","return_T5","return_T10"]].mean()
    ci.plot(kind="bar", ax=ax)
    ax.set_title("Avg Return by Asset Class")
    ax.set_ylabel("% Return"); ax.axhline(0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=0)

    # Top 15 most news-sensitive assets
    ax = axes[2,1]
    ai = df.groupby("asset")["return_T3"].apply(lambda x: x.abs().mean()).nlargest(15)
    ai.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Assets by Avg |Return| at T+3")
    ax.set_xlabel("Avg |% Return|")

    plt.tight_layout()
    plt.savefig("news_price_impact_v2_charts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Charts saved: news_price_impact_v2_charts.png")

# ── 11. SAVE RESULTS ──────────────────────────────────────────────────────────
def save_results(df, summary):
    out_cols = ["date","title","asset","asset_class","finbert_compound","finbert_pos",
                "finbert_neg","is_macro","news_count","impact_speed","max_abs_return",
                "return_T1","return_T3","return_T5","return_T10"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv("news_price_impact_v2_results.csv", index=False)
    pd.DataFrame(summary).T.to_csv("model_performance_v2.csv")
    df.groupby("asset").agg(
        n_events       =("title","count"),
        avg_sentiment  =("finbert_compound","mean"),
        avg_T1         =("return_T1","mean"),
        avg_T3         =("return_T3","mean"),
        avg_abs_T3     =("return_T3", lambda x: x.abs().mean()),
        avg_speed      =("impact_speed","mean"),
    ).round(4).to_csv("asset_impact_summary_v2.csv")
    print("Saved: news_price_impact_v2_results.csv, model_performance_v2.csv, asset_impact_summary_v2.csv")

# ── 12. PREDICT NEW HEADLINE ──────────────────────────────────────────────────
def predict_headline(headline: str, asset: str, date: str = None, prices_df=None) -> dict:
    """
    Predict how a new headline will affect an asset's price.

    Args:
        headline : the news title  e.g. "NVIDIA beats Q4 earnings, raises guidance"
        asset    : ticker          e.g. "NVDA", "BTC-USD", "GC=F"
        date     : "YYYY-MM-DD"   (defaults to today)
        prices_df: market prices DataFrame (for technical indicators, optional)

    Returns:
        dict of predictions for T+1, T+3, T+5, T+10 with:
          predicted_return_% | direction (UP/FLAT/DOWN) | confidence | prob_up/flat/down
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")
    date = pd.Timestamp(date)

    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    le_asset     = joblib.load(f"{MODEL_DIR}/le_asset.pkl")
    le_class     = joblib.load(f"{MODEL_DIR}/le_class.pkl")

    # FinBERT sentiment
    try:
        from transformers import pipeline
        pipe = pipeline("text-classification", model="ProsusAI/finbert",
                        device=-1, truncation=True, max_length=128, top_k=None)
        res  = pipe(headline)[0]
        s    = {r["label"]: r["score"] for r in res}
        fb_pos, fb_neg, fb_neu = s.get("positive",0), s.get("negative",0), s.get("neutral",0)
    except Exception:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        c = SentimentIntensityAnalyzer().polarity_scores(headline)["compound"]
        fb_pos, fb_neg, fb_neu = max(c,0), max(-c,0), 1-abs(c)
    fb_compound = fb_pos - fb_neg

    # Asset class encoding
    asset_class = ("crypto" if asset in CRYPTO_ASSETS else
                   "commodity" if asset in COMMODITY_ASSETS else "stock")
    asset_enc = int(le_asset.transform([asset])[0]) if asset in le_asset.classes_ else 0
    class_enc = int(le_class.transform([asset_class])[0])

    # Technical indicators
    tech = {}
    if prices_df is not None:
        technicals = compute_all_technicals(prices_df)
        tech = _lookup_technicals(technicals, asset, date)

    is_macro = int(any(kw in headline.lower() for kw in ASSET_KEYWORDS["_MACRO"]))

    row = {
        "finbert_pos": fb_pos, "finbert_neg": fb_neg,
        "finbert_neu": fb_neu, "finbert_compound": fb_compound,
        "day_of_week": date.dayofweek, "month": date.month,
        "is_macro": is_macro, "news_count": 1,
        "asset_enc": asset_enc, "asset_class_enc": class_enc,
    }
    for col in TECH_COLS:
        row[col] = tech.get(col, np.nan)

    X = pd.DataFrame([row])[feature_cols].fillna(0)

    label_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
    results = {
        "headline": headline, "asset": asset, "date": str(date.date()),
        "sentiment": "POSITIVE" if fb_compound > 0.1 else "NEGATIVE" if fb_compound < -0.1 else "NEUTRAL",
        "finbert_compound": round(fb_compound, 4),
        "predictions": {}
    }
    for h in HORIZONS:
        reg  = joblib.load(f"{MODEL_DIR}/reg_T{h}.pkl")
        clf  = joblib.load(f"{MODEL_DIR}/clf_T{h}.pkl")
        ret  = float(reg.predict(X)[0])
        probs = clf.predict_proba(X)[0]
        direction = label_map[int(clf.predict(X)[0])]
        results["predictions"][f"T+{h}"] = {
            "predicted_return_%":  round(ret, 3),
            "direction":           direction,
            "confidence":          round(float(max(probs)), 3),
            "prob_up":             round(float(probs[2]), 3),
            "prob_flat":           round(float(probs[1]), 3),
            "prob_down":           round(float(probs[0]), 3),
        }

    # Print nicely
    print(f"\nHeadline : {headline}")
    print(f"Asset    : {asset}")
    print(f"Sentiment: {results['sentiment']} ({fb_compound:+.3f})")
    print(f"{'Horizon':<10} {'Return%':>10} {'Direction':<10} {'Confidence':>12} {'P(up)':>8}")
    print("-" * 55)
    for k, v in results["predictions"].items():
        print(f"{k:<10} {v['predicted_return_%']:>9.2f}%  "
              f"{v['direction']:<10} {v['confidence']:>11.1%}  {v['prob_up']:>7.1%}")
    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NEWS -> PRICE IMPACT  v2")
    print("=" * 60)

    print("\n[1] Loading market data")
    prices = load_market_data(MARKET_PATH)

    print("\n[2] Computing technical indicators")
    technicals = compute_all_technicals(prices)

    print("\n[3] Loading news")
    news = load_news_data(NEWS_PATH)

    print("\n[4] FinBERT sentiment scoring")
    news = score_finbert(news)

    print("\n[5] Matching news to assets")
    news_assets = match_news_to_assets(news)

    print("\n[6] Adding news volume")
    news_assets = add_news_volume(news_assets)

    print("\n[7] Computing price impacts")
    impact_df = compute_price_impacts(news_assets, prices)

    print("\n[8] Building features")
    impact_df, feature_cols = build_features(impact_df, technicals)

    print("\n[9] Training models")
    summary, chart_data = train_models(impact_df, feature_cols)

    print("\n[10] Generating charts")
    make_charts(impact_df, chart_data, feature_cols)

    print("\n[11] Saving results")
    save_results(impact_df, summary)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    for horizon, m in summary.items():
        print(f"\n{horizon}:  MAE={m['MAE_%']}%  R2={m['R2']}  "
              f"CV_R2={m['CV_R2_mean']}+-{m['CV_R2_std']}  Acc={m['Clf_Accuracy']}")

    # Demo prediction
    print("\n" + "=" * 60)
    print("DEMO: predict_headline()")
    print("=" * 60)
    predict_headline(
        headline="NVIDIA beats earnings expectations and raises full-year guidance",
        asset="NVDA",
        date="2026-02-28",
        prices_df=prices
    )


if __name__ == "__main__":
    main()
