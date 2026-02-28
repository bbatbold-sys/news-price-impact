"""
News -> Price Impact ML Pipeline
- Loads GDELT news titles + dates
- Loads market data (stocks, commodities, crypto)
- Scores sentiment with VADER
- Maps news to relevant assets via keyword matching
- Computes price change at T+1, T+3, T+5, T+10 days after each news event
- Trains XGBoost model to predict price impact direction & magnitude
- Outputs results CSV + feature importance chart
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── 1. ASSET KEYWORD MAP ──────────────────────────────────────────────────────
# Maps asset ticker → list of keywords to search in headlines
ASSET_KEYWORDS = {
    # Stocks
    "NVDA": ["nvidia", "nvda", "jensen huang", "geforce", "cuda"],
    "TSLA": ["tesla", "tsla", "elon musk", "electric vehicle", "ev maker", "cybertruck"],
    "AAPL": ["apple", "aapl", "iphone", "ipad", "tim cook", "app store", "macbook"],
    "AMD":  ["amd", "advanced micro devices", "ryzen", "radeon"],
    "AMZN": ["amazon", "amzn", "aws", "jeff bezos", "prime", "whole foods"],
    "META": ["meta", "facebook", "instagram", "zuckerberg", "whatsapp", "threads"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "copilot", "satya nadella", "openai"],
    "GOOGL":["google", "googl", "alphabet", "youtube", "gemini", "sundar pichai"],
    "PLTR": ["palantir", "pltr"],
    "INTC": ["intel", "intc"],
    "BABA": ["alibaba", "baba", "jack ma", "ant group", "taobao"],
    "SOFI": ["sofi", "social finance"],
    "RIVN": ["rivian", "rivn"],
    "LCID": ["lucid", "lcid"],
    "SNAP": ["snapchat", "snap inc"],
    "SHOP": ["shopify", "shop"],
    "UBER": ["uber"],
    "NET":  ["cloudflare"],
    "COIN": ["coinbase", "coin"],
    "BAC":  ["bank of america", "bac"],
    "JPM":  ["jpmorgan", "jp morgan", "jamie dimon"],
    "C":    ["citigroup", "citibank", "citi"],
    "WFC":  ["wells fargo"],
    "MS":   ["morgan stanley"],
    "GS":   ["goldman sachs"],
    "V":    ["visa"],
    "MA":   ["mastercard"],
    "HOOD": ["robinhood"],
    "SCHW": ["charles schwab", "schwab"],
    "SPY":  ["s&p 500", "sp500", "spy etf"],
    "QQQ":  ["nasdaq", "qqq"],
    "IWM":  ["russell 2000", "small cap"],
    "PFE":  ["pfizer"],
    "MRNA": ["moderna", "mrna"],
    "JNJ":  ["johnson & johnson", "j&j"],
    "ABBV": ["abbvie"],
    "LLY":  ["eli lilly", "lilly"],
    "XOM":  ["exxon", "exxonmobil"],
    "CVX":  ["chevron"],
    "OXY":  ["occidental", "oxy petroleum"],
    "GE":   ["ge aerospace", "general electric"],
    "BA":   ["boeing"],
    "DIS":  ["disney", "dis"],
    "NFLX": ["netflix", "nflx"],
    "NIO":  ["nio"],
    "F":    ["ford motor", "ford f-"],
    "GME":  ["gamestop", "gme"],
    # Commodities
    "CL=F": ["crude oil", "wti", "oil price", "petroleum", "brent", "opec"],
    "BZ=F": ["brent crude", "brent oil"],
    "NG=F": ["natural gas", "lng"],
    "GC=F": ["gold", "gold price", "bullion"],
    "SI=F": ["silver", "silver price"],
    "ZC=F": ["corn", "corn price", "grain"],
    "ZS=F": ["soybean", "soy"],
    "ZW=F": ["wheat", "wheat price"],
    "HG=F": ["copper", "copper price"],
    "KC=F": ["coffee", "arabica"],
    # Crypto
    "BTC-USD": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "ETH-USD": ["ethereum", "eth", "ether"],
    "USDT-USD":["tether", "usdt", "stablecoin"],
    "SOL-USD": ["solana", "sol"],
    "XRP-USD": ["xrp", "ripple"],
    # Broad market / macro news applies to all
    "_MACRO": ["fed", "federal reserve", "interest rate", "inflation", "recession",
               "gdp", "unemployment", "cpi", "tariff", "trade war", "debt ceiling",
               "rate hike", "rate cut", "jerome powell", "treasury", "fiscal"],
}

MACRO_ASSETS = ["NVDA","TSLA","AAPL","AMD","AMZN","META","MSFT","GOOGL",
                "SPY","QQQ","BTC-USD","ETH-USD","GC=F","CL=F"]

# ── 2. LOAD MARKET DATA ───────────────────────────────────────────────────────
def load_market_data(path):
    xl = pd.ExcelFile(path)
    frames = {}
    for sheet in ["Stocks", "Commodities", "Crypto"]:
        df = xl.parse(sheet, header=None)
        # Row 0 is title, row 1 is header
        df.columns = df.iloc[1]
        df = df.iloc[2:].reset_index(drop=True)
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        # Clean column names: take just the ticker (before \n)
        new_cols = {"Date": "Date"}
        for c in df.columns[1:]:
            if pd.notna(c) and "\n" in str(c):
                ticker = str(c).split("\n")[0].strip()
                new_cols[c] = ticker
            elif pd.notna(c):
                new_cols[c] = str(c).strip()
        df = df.rename(columns=new_cols)
        # Convert price columns to numeric
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("Date").sort_index()
        frames[sheet] = df
    # Merge all into one wide DataFrame
    all_prices = pd.concat([frames["Stocks"], frames["Commodities"], frames["Crypto"]], axis=1)
    all_prices = all_prices.loc[~all_prices.index.duplicated(keep="first")]
    print(f"Market data: {len(all_prices)} trading days, {len(all_prices.columns)} assets")
    return all_prices

# ── 3. LOAD NEWS DATA ─────────────────────────────────────────────────────────
def load_news_data(path):
    df = pd.read_csv(path)
    df = df[["title", "seendate"]].dropna()
    df["date"] = pd.to_datetime(df["seendate"].str[:8], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "title"])
    df["title_lower"] = df["title"].str.lower()
    print(f"News data: {len(df)} articles, {df['date'].min().date()} to {df['date'].max().date()}")
    return df

# ── 4. SENTIMENT SCORING ──────────────────────────────────────────────────────
def score_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    scores = df["title"].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    df = df.copy()
    df["sentiment"] = scores
    df["sentiment_label"] = pd.cut(scores, bins=[-1.01, -0.05, 0.05, 1.01],
                                    labels=["negative", "neutral", "positive"])
    return df

# ── 5. MAP NEWS TO ASSETS ─────────────────────────────────────────────────────
def match_news_to_assets(df):
    rows = []
    for _, row in df.iterrows():
        title_l = row["title_lower"]
        matched = []
        is_macro = any(kw in title_l for kw in ASSET_KEYWORDS["_MACRO"])
        for ticker, keywords in ASSET_KEYWORDS.items():
            if ticker == "_MACRO":
                continue
            if any(kw in title_l for kw in keywords):
                matched.append(ticker)
        if is_macro and not matched:
            matched = MACRO_ASSETS
        elif is_macro:
            # also add macro assets
            matched = list(set(matched + MACRO_ASSETS))
        if not matched:
            continue
        for asset in matched:
            rows.append({**row.to_dict(), "asset": asset, "is_macro": is_macro})
    result = pd.DataFrame(rows)
    print(f"News-asset pairs: {len(result)} ({result['asset'].nunique()} unique assets)")
    return result

# ── 6. COMPUTE PRICE IMPACT ───────────────────────────────────────────────────
def get_future_return(prices, asset, event_date, horizon_days):
    """Get % price change from event_date to event_date + horizon trading days."""
    if asset not in prices.columns:
        return np.nan
    future_dates = prices.index[prices.index > event_date]
    if len(future_dates) < horizon_days:
        return np.nan
    target_date = future_dates[horizon_days - 1]
    # Price on or just before event date
    past_dates = prices.index[prices.index <= event_date]
    if len(past_dates) == 0:
        return np.nan
    p0 = prices.loc[past_dates[-1], asset]
    p1 = prices.loc[target_date, asset]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return (p1 - p0) / p0 * 100  # % change

def compute_price_impacts(news_assets, prices, horizons=[1, 3, 5, 10]):
    print("Computing price impacts (this may take a minute)...")
    results = []
    total = len(news_assets)
    for i, (_, row) in enumerate(news_assets.iterrows()):
        if i % 5000 == 0:
            print(f"  {i}/{total}")
        record = {
            "date": row["date"],
            "title": row["title"],
            "asset": row["asset"],
            "sentiment": row["sentiment"],
            "sentiment_label": row["sentiment_label"],
            "is_macro": row["is_macro"],
        }
        for h in horizons:
            record[f"return_T{h}"] = get_future_return(prices, row["asset"], row["date"], h)
        # Impact speed: first horizon where |return| > 1%
        speed = None
        for h in horizons:
            r = record.get(f"return_T{h}", np.nan)
            if pd.notna(r) and abs(r) > 1.0:
                speed = h
                break
        record["impact_speed_days"] = speed
        record["max_return"] = max(
            [abs(record.get(f"return_T{h}", 0) or 0) for h in horizons]
        )
        results.append(record)
    df = pd.DataFrame(results).dropna(subset=["return_T1"])
    print(f"Final dataset: {len(df)} news-asset-impact rows")
    return df

# ── 7. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek      # 0=Mon
    df["month"] = df["date"].dt.month
    df["is_macro"] = df["is_macro"].astype(int)

    # Asset class
    crypto_assets = {"BTC-USD", "ETH-USD", "USDT-USD", "SOL-USD", "XRP-USD"}
    commodity_assets = {"CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "ZC=F", "ZS=F", "ZW=F", "HG=F", "KC=F"}
    df["asset_class"] = df["asset"].apply(
        lambda a: "crypto" if a in crypto_assets else ("commodity" if a in commodity_assets else "stock")
    )

    le_asset = LabelEncoder()
    le_class = LabelEncoder()
    df["asset_enc"] = le_asset.fit_transform(df["asset"])
    df["asset_class_enc"] = le_class.fit_transform(df["asset_class"])

    feature_cols = ["sentiment", "day_of_week", "month", "is_macro",
                    "asset_enc", "asset_class_enc"]
    return df, feature_cols, le_asset, le_class

# ── 8. TRAIN XGBoost MODELS ───────────────────────────────────────────────────
def train_models(df, feature_cols):
    results_summary = {}
    charts_data = {}

    for horizon in [1, 3, 5, 10]:
        target = f"return_T{horizon}"
        sub = df[feature_cols + [target, "asset", "asset_class"]].dropna(subset=[target])
        X = sub[feature_cols]
        y = sub[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- Regression: predict % price change ---
        reg = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbosity=0)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)

        # --- Classification: Up / Down / Flat ---
        y_cls_raw = pd.cut(y, bins=[-np.inf, -1, 1, np.inf], labels=[0, 1, 2]).astype(int)
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_cls_raw, test_size=0.2, random_state=42
        )
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0,
                                  eval_metric="mlogloss", num_class=3)
        clf.fit(X_train_c, y_train_c)
        clf_acc = (clf.predict(X_test_c) == y_test_c).mean()

        results_summary[f"T+{horizon}"] = {
            "MAE (%)": round(mae, 4),
            "R2": round(r2, 4),
            "Classifier Accuracy": round(clf_acc, 4),
            "n_samples": len(sub),
        }
        charts_data[f"T+{horizon}"] = {
            "reg": reg, "clf": clf, "X_test": X_test,
            "y_test": y_test, "y_pred": y_pred,
            "importances": reg.feature_importances_,
            "features": feature_cols,
        }
        print(f"  T+{horizon}: MAE={mae:.3f}%  R2={r2:.4f}  Clf Acc={clf_acc:.3f}")

    return results_summary, charts_data

# ── 9. CHARTS ─────────────────────────────────────────────────────────────────
def make_charts(df, charts_data, feature_cols):
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("News Impact on Asset Prices (GDELT + Yahoo Finance)", fontsize=16, fontweight="bold")

    # --- Sentiment distribution ---
    ax = axes[0, 0]
    df["sentiment_label"].value_counts().plot(kind="bar", ax=ax, color=["red","gray","green"])
    ax.set_title("News Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    # --- Avg return by sentiment at T+1 and T+3 ---
    ax = axes[0, 1]
    sent_impact = df.groupby("sentiment_label")[["return_T1", "return_T3"]].mean()
    sent_impact.plot(kind="bar", ax=ax, color=["steelblue","darkorange"])
    ax.set_title("Avg Price Change by Sentiment")
    ax.set_ylabel("% Return")
    ax.tick_params(axis="x", rotation=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(["T+1 day", "T+3 days"])

    # --- Feature importance for T+1 ---
    ax = axes[1, 0]
    imp = charts_data["T+1"]["importances"]
    feat_names = [f.replace("_enc", "").replace("_", " ") for f in feature_cols]
    sorted_idx = np.argsort(imp)
    ax.barh([feat_names[i] for i in sorted_idx], imp[sorted_idx], color="steelblue")
    ax.set_title("Feature Importance (T+1 Regression)")
    ax.set_xlabel("Importance Score")

    # --- Impact speed histogram ---
    ax = axes[1, 1]
    speed = df["impact_speed_days"].dropna()
    ax.hist(speed, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            color="darkorange", edgecolor="black")
    ax.set_title("Impact Speed Distribution (days to >1% move)")
    ax.set_xlabel("Days after news")
    ax.set_ylabel("Count")

    # --- Avg T+1 return by asset class ---
    ax = axes[2, 0]
    cls_impact = df.groupby("asset_class")[["return_T1","return_T3","return_T5"]].mean()
    cls_impact.plot(kind="bar", ax=ax)
    ax.set_title("Avg Return by Asset Class")
    ax.set_ylabel("% Return")
    ax.tick_params(axis="x", rotation=0)
    ax.axhline(0, color="black", linewidth=0.8)

    # --- Top 15 most impacted assets (avg |return| at T+3) ---
    ax = axes[2, 1]
    asset_impact = df.groupby("asset")["return_T3"].apply(lambda x: x.abs().mean()).nlargest(15)
    asset_impact.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Assets by Avg |Return| at T+3")
    ax.set_xlabel("Avg |% Return|")

    plt.tight_layout()
    out = "news_price_impact_charts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts saved: {out}")

# ── 10. SAVE RESULTS ──────────────────────────────────────────────────────────
def save_results(df, results_summary):
    # Full detailed results
    out_cols = ["date", "title", "asset", "asset_class", "sentiment",
                "sentiment_label", "is_macro", "impact_speed_days", "max_return",
                "return_T1", "return_T3", "return_T5", "return_T10"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv("news_price_impact_results.csv", index=False)

    # Model performance summary
    pd.DataFrame(results_summary).T.to_csv("model_performance.csv")

    # Per-asset average impact summary
    asset_summary = df.groupby("asset").agg(
        n_events=("title", "count"),
        avg_sentiment=("sentiment", "mean"),
        avg_return_T1=("return_T1", "mean"),
        avg_return_T3=("return_T3", "mean"),
        avg_return_T5=("return_T5", "mean"),
        avg_abs_return_T3=("return_T3", lambda x: x.abs().mean()),
        avg_impact_speed=("impact_speed_days", "mean"),
    ).round(4)
    asset_summary.to_csv("asset_impact_summary.csv")
    print("Results saved:")
    print("  news_price_impact_results.csv  (full detail)")
    print("  model_performance.csv          (ML model metrics)")
    print("  asset_impact_summary.csv       (per-asset averages)")
    print("  news_price_impact_charts.png   (charts)")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    MARKET_PATH = r"C:\Users\batbo\OneDrive\Desktop\Market_Data_5Years.xlsx"
    NEWS_PATH   = r"C:\Users\batbo\gdelt_yahoo_finance.csv"

    print("=" * 60)
    print("Step 1: Loading market data")
    prices = load_market_data(MARKET_PATH)

    print("\nStep 2: Loading news data")
    news = load_news_data(NEWS_PATH)

    print("\nStep 3: Scoring sentiment")
    news = score_sentiment(news)

    print("\nStep 4: Matching news to assets")
    news_assets = match_news_to_assets(news)

    print("\nStep 5: Computing price impacts")
    impact_df = compute_price_impacts(news_assets, prices)

    print("\nStep 6: Building features")
    impact_df, feature_cols, le_asset, le_class = build_features(impact_df)

    print("\nStep 7: Training XGBoost models")
    results_summary, charts_data = train_models(impact_df, feature_cols)

    print("\nStep 8: Generating charts")
    make_charts(impact_df, charts_data, feature_cols)

    print("\nStep 9: Saving results")
    save_results(impact_df, results_summary)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    for horizon, metrics in results_summary.items():
        print(f"\n{horizon}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print("\nDone. Check C:\\Users\\batbo\\ for output files.")

if __name__ == "__main__":
    main()
