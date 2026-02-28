# News → Price Impact ML Model

Measures **how much** and **how fast** Yahoo Finance news articles affect the closing prices of stocks, commodities, and cryptocurrencies.

## What it does

1. **Fetches news** from GDELT — 2 years of articles published on `finance.yahoo.com`
2. **Scores sentiment** on every headline using VADER (positive / neutral / negative)
3. **Maps each news article** to relevant assets via keyword matching (50 stocks, 10 commodities, 5 cryptos)
4. **Computes price impact** at T+1, T+3, T+5, T+10 trading days after each article
5. **Trains XGBoost models** to predict the direction and magnitude of price moves

## Results

| Horizon | Avg Error | Direction Accuracy |
|---------|-----------|-------------------|
| T+1 day | ±1.53% | 58.1% |
| T+3 days | ±2.89% | 57.3% |
| T+5 days | ±4.23% | 62.4% |
| T+10 days | ±4.86% | 62.2% |

![Charts](news_price_impact_charts.png)

## Assets covered

**Stocks:** NVDA, TSLA, AAPL, AMD, AMZN, META, MSFT, GOOGL, PLTR, INTC, BABA, JPM, BAC, GS, MS, V, MA, NFLX, DIS, COIN, HOOD, SPY, QQQ, and more

**Commodities:** Crude Oil, Brent, Natural Gas, Gold, Silver, Corn, Soybeans, Wheat, Copper, Coffee

**Crypto:** BTC, ETH, SOL, XRP, USDT

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1 — Fetch news from GDELT
```bash
python gdelt_yahoo_finance.py
```
Outputs `gdelt_yahoo_finance.csv` (~14k articles, 2024–2026).

### Step 2 — Run the ML pipeline
```bash
python news_price_impact.py
```
Requires:
- `gdelt_yahoo_finance.csv` (from Step 1)
- `Market_Data_5Years.xlsx` (your market data file with Stocks / Commodities / Crypto sheets)

Outputs:
- `news_price_impact_results.csv` — every news event + price change at each horizon
- `asset_impact_summary.csv` — per-asset averages
- `model_performance.csv` — ML model metrics
- `news_price_impact_charts.png` — visualizations

## How it works

```
GDELT API → News titles + dates
                ↓
         VADER Sentiment Score (-1 to +1)
                ↓
    Keyword matching → asset ticker(s)
                ↓
    Price change at T+1, T+3, T+5, T+10
                ↓
    XGBoost Regressor  →  predicted % change
    XGBoost Classifier →  Up / Flat / Down
```

## Data sources

- **News:** [GDELT Project](https://www.gdeltproject.org/) via Doc API
- **Prices:** Yahoo Finance (adjusted close prices)
