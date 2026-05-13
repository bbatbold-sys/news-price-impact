"""PULSE — News Impact Terminal  (port 5001)"""
import os, sys, json, logging, threading
from datetime import datetime, timezone
from queue import Queue, Empty
import requests as _req
from flask import Flask, render_template, jsonify, Response, stream_with_context, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("pulse")

_DIR         = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_DIR, "..")
SIGNALS_FILE = os.path.join(_ROOT, "web", "signals_export.json")
PRICES_FILE  = os.path.join(_ROOT, "web", "prices_export.json")
EXTREME_FILE = os.path.join(_ROOT, "web", "extreme_alerts.json")

MARKET_CANDIDATES = [
    r"C:\Users\batbo\Desktop\Market_Data_5Years.xlsx",
    r"C:\Users\batbo\OneDrive\Desktop\Market_Data_5Years.xlsx",
    r"C:\Users\batbo\Downloads\Market_Data_5Years.xlsx",
]

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

_YF_HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_price_cache : dict = {}
_price_lock  = threading.Lock()
_price_subs  : list = []
_subs_lock   = threading.Lock()

# ── Optional predictor (loads in background) ──────────────────────────────────
_predict_fn      = None
_predictor_ready = False

def _init_predictor():
    global _predict_fn, _predictor_ready
    sys.path.insert(0, os.path.join(_DIR, "..", "web"))
    try:
        import predictor as _p
        mkt = next((p for p in MARKET_CANDIDATES if os.path.exists(p)), None)
        if mkt:
            _p.load_market(mkt)
        _predict_fn      = _p.predict
        _predictor_ready = True
        log.info("Predictor ready")
    except Exception as e:
        log.warning(f"Predictor unavailable: {e}")

threading.Thread(target=_init_predictor, daemon=True).start()

# ── Price fetching ─────────────────────────────────────────────────────────────
def _fetch_prices(symbols):
    result = {}
    for i in range(0, len(symbols), 20):
        batch = ",".join(symbols[i:i+20])
        try:
            url = (f"https://query1.finance.yahoo.com/v7/finance/quote"
                   f"?symbols={batch}&fields=regularMarketPrice,regularMarketPreviousClose")
            r = _req.get(url, headers=_YF_HEADERS, timeout=15)
            for q in r.json().get("quoteResponse", {}).get("result", []):
                sym = q.get("symbol","")
                p   = q.get("regularMarketPrice")
                pv  = q.get("regularMarketPreviousClose")
                if sym and p and pv:
                    result[sym] = {"price":  round(float(p), 4),
                                   "change": round((float(p)-float(pv))/float(pv)*100, 2)}
        except Exception as e:
            log.debug(f"YF error: {e}")
    return result

def _refresh_prices():
    all_syms = [s for v in ALL_ASSETS.values() for s in v]
    new = _fetch_prices(all_syms)
    if not new:
        try:
            with open(PRICES_FILE) as f: new = json.load(f)
        except Exception: pass
    if new:
        with _price_lock:
            _price_cache.update(new)
        msg = "data: " + json.dumps(dict(_price_cache)) + "\n\n"
        with _subs_lock:
            dead = []
            for q in _price_subs:
                try: q.put_nowait(msg)
                except: dead.append(q)
            for q in dead: _price_subs.remove(q)
        log.info(f"Prices: {len(new)}/{len(all_syms)}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_signals():
    """Load latest signal per asset — prefer live poller data, fall back to export file."""
    from light_poller import get_live_news
    live = get_live_news()
    if live:
        # Build latest-per-asset from live news
        by_asset = {}
        for row in live:
            a = row.get("asset")
            if a and a not in by_asset:
                by_asset[a] = row
        # Fill in any assets missing from live with signals_export fallback
        try:
            with open(SIGNALS_FILE) as f:
                for row in json.load(f):
                    a = row.get("asset")
                    if a and a not in by_asset:
                        by_asset[a] = row
        except Exception:
            pass
        return list(by_asset.values())
    # Full fallback to export file
    try:
        with open(SIGNALS_FILE) as f: return json.load(f)
    except Exception: return []

def fmt_seconds(s):
    if s is None: return "—"
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    h = s//3600; m = (s%3600)//60
    return (f"{h}h {m}m" if m else f"{h}h")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/signals")
def api_signals():
    signals  = load_signals()
    by_asset = {s["asset"]: s for s in signals}
    rows = []
    for cls, assets in ALL_ASSETS.items():
        for asset in assets:
            sig = by_asset.get(asset)
            row = {"asset": asset, "name": ASSET_NAMES.get(asset, asset), "asset_class": cls}
            if sig:
                row.update({
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
            rows.append(row)
    return jsonify({"signals": rows, "updated": datetime.now(timezone.utc).isoformat()})

@app.route("/api/news")
def api_news():
    from light_poller import get_live_news
    live = get_live_news()
    source = live if live else load_signals()
    news = sorted(source, key=lambda x: x.get("detected_at",""), reverse=True)[:60]
    for n in news: n["time_to_react"] = fmt_seconds(n.get("seconds_to_react"))
    return jsonify({"news": news, "count": len(news)})

@app.route("/api/prices")
def api_prices():
    with _price_lock: cache = dict(_price_cache)
    if not cache:
        try:
            with open(PRICES_FILE) as f: cache = json.load(f)
        except Exception: pass
    return jsonify({"prices": cache})

@app.route("/api/prices/stream")
def price_stream():
    q = Queue(maxsize=10)
    with _subs_lock: _price_subs.append(q)
    def generate():
        try:
            with _price_lock: snap = dict(_price_cache)
            if snap: yield "data: " + json.dumps(snap) + "\n\n"
            while True:
                try:    yield q.get(timeout=25)
                except Empty: yield ": keepalive\n\n"
        except GeneratorExit: pass
        finally:
            with _subs_lock:
                try: _price_subs.remove(q)
                except ValueError: pass
    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data     = request.json or {}
    headline = data.get("headline","").strip()
    asset    = data.get("asset","NVDA")
    if not headline:
        return jsonify({"error":"headline required"}), 400
    if not _predictor_ready or not _predict_fn:
        return jsonify({"error":"Predictor still loading — try again in 30s"}), 503
    try:
        return jsonify(_predict_fn(headline, asset))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/extreme")
def api_extreme():
    try:
        with open(EXTREME_FILE) as f: return jsonify({"alerts": json.load(f)})
    except Exception: return jsonify({"alerts": []})

# ── Start ─────────────────────────────────────────────────────────────────────
import light_poller
light_poller.start()
_refresh_prices()
scheduler = BackgroundScheduler()
scheduler.add_job(_refresh_prices, "interval", seconds=60)
scheduler.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    log.info(f"PULSE → http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
