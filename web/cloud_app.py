"""
Lightweight cloud dashboard for Render.com.
No ML, no poller — reads signals from signals_export.json (committed to GitHub).
Fetches live prices from yfinance.
"""
import os, sys, json, logging, threading
from datetime import datetime, timezone
from queue import Queue, Empty
from flask import Flask, render_template, jsonify, Response, stream_with_context
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("cloud_app")

WEB_DIR       = os.path.dirname(os.path.abspath(__file__))
EXPORT_FILE   = os.path.join(WEB_DIR, "signals_export.json")
PRICES_FILE   = os.path.join(WEB_DIR, "prices_export.json")

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

# ── Load signals from JSON export ─────────────────────────────────────────────
def load_signals() -> list:
    try:
        with open(EXPORT_FILE) as f:
            return json.load(f)
    except Exception:
        return []

# ── Live price streaming ───────────────────────────────────────────────────────
_price_cache: dict = {}
_price_lock  = threading.Lock()
_price_subs: list = []
_subs_lock   = threading.Lock()

import requests as _requests

_YF_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def _push_prices(prices: dict):
    msg = "data: " + json.dumps(prices) + "\n\n"
    with _subs_lock:
        dead = []
        for q in _price_subs:
            try:    q.put_nowait(msg)
            except: dead.append(q)
        for q in dead:
            _price_subs.remove(q)

def _fetch_prices_yf(symbols: list) -> dict:
    """Fetch prices via Yahoo Finance quote API — no yfinance package needed."""
    result = {}
    batch = ",".join(symbols)
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={batch}&fields=regularMarketPrice,regularMarketPreviousClose"
        r = _requests.get(url, headers=_YF_HEADERS, timeout=15)
        data = r.json()
        for q in data.get("quoteResponse", {}).get("result", []):
            sym   = q.get("symbol", "")
            price = q.get("regularMarketPrice")
            prev  = q.get("regularMarketPreviousClose")
            if sym and price and prev:
                result[sym] = {
                    "price":  round(float(price), 4),
                    "change": round((float(price) - float(prev)) / float(prev) * 100, 2),
                }
    except Exception as e:
        log.debug(f"YF API error: {e}")
    return result

def _refresh_prices():
    """Fetch live prices from Yahoo Finance API, fall back to file."""
    all_syms = [s for v in ALL_ASSETS.values() for s in v]
    new = {}
    # Fetch in batches of 20
    for i in range(0, len(all_syms), 20):
        new.update(_fetch_prices_yf(all_syms[i:i+20]))
    # Fall back to file if API returned nothing
    if not new:
        try:
            with open(PRICES_FILE) as f:
                new = json.load(f)
            log.info("Prices loaded from fallback file")
        except Exception:
            pass
    if new:
        with _price_lock:
            _price_cache.update(new)
        _push_prices(dict(_price_cache))
        log.info(f"Prices refreshed: {len(new)} tickers")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

def fmt_seconds(s):
    if s is None: return "Unknown"
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    h = s // 3600; m = (s % 3600) // 60
    return f"{h}h {m}m" if m else f"{h}h"

@app.route("/")
def index():
    try:
        with open(PRICES_FILE) as f:
            initial_prices = json.load(f)
    except Exception:
        initial_prices = {}
    return render_template("index.html", all_assets=ALL_ASSETS,
                           asset_names=ASSET_NAMES,
                           initial_prices=initial_prices)

@app.route("/api/signals")
def api_signals():
    signals = load_signals()
    by_asset = {s["asset"]: s for s in signals}
    rows = []
    for cls, assets in ALL_ASSETS.items():
        for asset in assets:
            sig = by_asset.get(asset)
            if sig:
                rows.append({
                    "asset":         asset,
                    "name":          ASSET_NAMES.get(asset, asset),
                    "asset_class":   cls,
                    "headline":      sig.get("headline"),
                    "sentiment":     sig.get("sentiment"),
                    "direction_T1":  sig.get("direction_T1"),
                    "direction_T3":  sig.get("direction_T3"),
                    "direction_T5":  sig.get("direction_T5"),
                    "pred_T1":       sig.get("pred_return_T1"),
                    "pred_T3":       sig.get("pred_return_T3"),
                    "pred_T5":       sig.get("pred_return_T5"),
                    "confidence_T3": sig.get("confidence_T3"),
                    "prob_up":       sig.get("prob_up_T3"),
                    "prob_down":     sig.get("prob_down_T3"),
                    "secs_to_react": sig.get("seconds_to_react"),
                    "time_to_react": fmt_seconds(sig.get("seconds_to_react")),
                    "detected_at":   sig.get("detected_at"),
                    "finbert":       sig.get("finbert_compound"),
                })
            else:
                rows.append({"asset": asset, "name": ASSET_NAMES.get(asset, asset),
                             "asset_class": cls, "headline": None})
    return jsonify({"signals": rows, "updated": datetime.now(timezone.utc).isoformat()})

@app.route("/api/news")
def api_news():
    signals = load_signals()
    news = sorted(signals, key=lambda x: x.get("detected_at",""), reverse=True)[:60]
    for n in news:
        n["time_to_react"] = fmt_seconds(n.get("seconds_to_react"))
    return jsonify({"news": news})

@app.route("/api/prices")
def api_prices():
    with _price_lock:
        cache = dict(_price_cache)
    if not cache:
        try:
            with open(PRICES_FILE) as f:
                cache = json.load(f)
        except Exception:
            pass
    return jsonify({"prices": cache})

@app.route("/api/prices/stream")
def price_stream():
    q = Queue(maxsize=10)
    with _subs_lock:
        _price_subs.append(q)
    def generate():
        try:
            with _price_lock:
                snap = dict(_price_cache)
            if snap:
                yield "data: " + json.dumps(snap) + "\n\n"
            while True:
                try:
                    yield q.get(timeout=25)
                except Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _subs_lock:
                try: _price_subs.remove(q)
                except ValueError: pass
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Load prices immediately at startup (synchronous) ─────────────────────────
_refresh_prices()

# ── Scheduler: reload prices every 60s ────────────────────────────────────────
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(_refresh_prices, "interval", seconds=60)
scheduler.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
