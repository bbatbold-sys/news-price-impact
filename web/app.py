"""
Flask web app — real-time news impact dashboard.
Run: python app.py
"""
import os, sys, logging
from datetime import datetime, timezone
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("app")

# ── Init DB ───────────────────────────────────────────────────────────────────
from db import init_db, latest_per_asset, recent_news
init_db()

# ── Load models & market data ─────────────────────────────────────────────────
import predictor
MARKET_PATH = os.path.join(os.path.dirname(__file__), "..",
              r"C:\Users\batbo\OneDrive\Desktop\Market_Data_5Years.xlsx"
              if os.path.exists(r"C:\Users\batbo\OneDrive\Desktop\Market_Data_5Years.xlsx")
              else "Market_Data_5Years.xlsx")
predictor.load_market(r"C:\Users\batbo\OneDrive\Desktop\Market_Data_5Years.xlsx")

# ── All assets we track ───────────────────────────────────────────────────────
ALL_ASSETS = {
    "stock": ["NVDA","TSLA","AAPL","AMD","AMZN","META","MSFT","GOOGL","PLTR",
              "INTC","BABA","SOFI","RIVN","LCID","SNAP","SHOP","UBER","NET",
              "COIN","BAC","JPM","C","WFC","MS","GS","V","MA","HOOD","SCHW",
              "SPY","QQQ","IWM","PFE","MRNA","JNJ","ABBV","LLY","XOM","CVX",
              "OXY","GE","BA","DIS","NFLX","NIO","F","GME"],
    "commodity": ["CL=F","BZ=F","NG=F","GC=F","SI=F","ZC=F","ZS=F","ZW=F","HG=F","KC=F"],
    "crypto":    ["BTC-USD","ETH-USD","USDT-USD","SOL-USD","XRP-USD"],
}

ASSET_NAMES = {
    "NVDA":"NVIDIA","TSLA":"Tesla","AAPL":"Apple","AMD":"AMD","AMZN":"Amazon",
    "META":"Meta","MSFT":"Microsoft","GOOGL":"Alphabet","PLTR":"Palantir",
    "INTC":"Intel","BABA":"Alibaba","SOFI":"SoFi","RIVN":"Rivian",
    "LCID":"Lucid","SNAP":"Snap","SHOP":"Shopify","UBER":"Uber",
    "NET":"Cloudflare","COIN":"Coinbase","BAC":"Bank of America",
    "JPM":"JPMorgan","C":"Citigroup","WFC":"Wells Fargo",
    "MS":"Morgan Stanley","GS":"Goldman Sachs","V":"Visa","MA":"Mastercard",
    "HOOD":"Robinhood","SCHW":"Schwab","SPY":"S&P 500 ETF","QQQ":"Nasdaq ETF",
    "IWM":"Russell 2000","PFE":"Pfizer","MRNA":"Moderna","JNJ":"J&J",
    "ABBV":"AbbVie","LLY":"Eli Lilly","XOM":"ExxonMobil","CVX":"Chevron",
    "OXY":"Occidental","GE":"GE Aerospace","BA":"Boeing","DIS":"Disney",
    "NFLX":"Netflix","NIO":"NIO","F":"Ford","GME":"GameStop",
    "CL=F":"Crude Oil","BZ=F":"Brent Crude","NG=F":"Natural Gas",
    "GC=F":"Gold","SI=F":"Silver","ZC=F":"Corn","ZS=F":"Soybeans",
    "ZW=F":"Wheat","HG=F":"Copper","KC=F":"Coffee",
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","USDT-USD":"Tether",
    "SOL-USD":"Solana","XRP-USD":"XRP",
}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

def fmt_seconds(s):
    if s is None:
        return "Unknown"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m" if m else f"{h}h"

@app.route("/")
def index():
    return render_template("index.html", all_assets=ALL_ASSETS,
                           asset_names=ASSET_NAMES)

@app.route("/api/signals")
def api_signals():
    signals = latest_per_asset()
    # Build a dict keyed by asset for fast lookup
    by_asset = {s["asset"]: s for s in signals}

    rows = []
    for cls, assets in ALL_ASSETS.items():
        for asset in assets:
            sig = by_asset.get(asset)
            if sig:
                rows.append({
                    "asset":          asset,
                    "name":           ASSET_NAMES.get(asset, asset),
                    "asset_class":    cls,
                    "headline":       sig["headline"],
                    "sentiment":      sig["sentiment"],
                    "direction_T1":   sig["direction_T1"],
                    "direction_T3":   sig["direction_T3"],
                    "direction_T5":   sig["direction_T5"],
                    "pred_T1":        sig["pred_return_T1"],
                    "pred_T3":        sig["pred_return_T3"],
                    "pred_T5":        sig["pred_return_T5"],
                    "confidence_T3":  sig["confidence_T3"],
                    "prob_up":        sig["prob_up_T3"],
                    "prob_down":      sig["prob_down_T3"],
                    "secs_to_react":  sig["seconds_to_react"],
                    "time_to_react":  fmt_seconds(sig["seconds_to_react"]),
                    "detected_at":    sig["detected_at"],
                    "finbert":        sig["finbert_compound"],
                })
            else:
                rows.append({
                    "asset":       asset,
                    "name":        ASSET_NAMES.get(asset, asset),
                    "asset_class": cls,
                    "headline":    None,
                })
    return jsonify({"signals": rows,
                    "updated": datetime.now(timezone.utc).isoformat()})

@app.route("/api/news")
def api_news():
    news = recent_news(60)
    for n in news:
        n["time_to_react"] = fmt_seconds(n.get("seconds_to_react"))
    return jsonify({"news": news})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    from flask import request
    data     = request.json or {}
    headline = data.get("headline", "")
    asset    = data.get("asset", "NVDA")
    if not headline:
        return jsonify({"error": "headline required"}), 400
    result = predictor.predict(headline, asset)
    result["time_to_react"] = fmt_seconds(
        __import__("poller").seconds_to_react(result["asset_class"])
    )
    return jsonify(result)

# ── Background poller ─────────────────────────────────────────────────────────
from poller import poll

scheduler = BackgroundScheduler()
scheduler.add_job(poll, "interval", minutes=5, id="gdelt_poll",
                  next_run_time=datetime.now())   # run immediately on start
scheduler.start()

if __name__ == "__main__":
    log.info("Starting News Impact Dashboard on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
