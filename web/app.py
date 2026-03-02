"""
Flask web app — real-time news impact dashboard.
Run: python app.py
"""
import os, sys, logging, threading, json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from datetime import datetime, timezone
from flask import Flask, render_template, jsonify, Response, stream_with_context
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

# ── Live price streaming (SSE) ────────────────────────────────────────────────
_price_cache: dict = {}
_price_lock  = threading.Lock()
_price_subs: list = []
_subs_lock   = threading.Lock()

def _push_prices(prices: dict):
    """Push latest prices to all connected SSE clients."""
    msg = "data: " + json.dumps(prices) + "\n\n"
    with _subs_lock:
        dead = []
        for q in _price_subs:
            try:    q.put_nowait(msg)
            except: dead.append(q)
        for q in dead:
            _price_subs.remove(q)

def _fetch_one(sym: str):
    import yfinance as yf
    try:
        fi    = yf.Ticker(sym).fast_info
        price = fi.last_price
        prev  = fi.previous_close
        if price and prev:
            p, c = float(price), float(prev)
            return sym, {"price": round(p, 4), "change": round((p - c) / c * 100, 2)}
    except Exception:
        pass
    return sym, None

def _refresh_prices():
    all_syms = [s for v in ALL_ASSETS.values() for s in v]
    new = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, data in zip(all_syms, ex.map(_fetch_one, all_syms)):
            if data[1]:
                new[data[0]] = data[1]
    if new:
        with _price_lock:
            _price_cache.update(new)
        _push_prices(dict(_price_cache))
        log.info(f"Prices pushed: {len(new)}/{len(all_syms)} tickers")
        # Export prices to file for Render cloud dashboard
        try:
            import json as _json, os as _os
            _prices_file = _os.path.join(_os.path.dirname(__file__), "..", "prices_export.json")
            with open(_prices_file, "w") as _f:
                _json.dump(dict(_price_cache), _f)
        except Exception:
            pass

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

@app.route("/api/prices")
def api_prices():
    with _price_lock:
        return jsonify({"prices": dict(_price_cache)})

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
scheduler.add_job(poll, "interval", minutes=1, id="yf_poll",
                  next_run_time=datetime.now())   # run immediately on start
scheduler.add_job(_refresh_prices, "interval", seconds=30, id="price_refresh",
                  next_run_time=datetime.now())   # real-time prices every 30s
scheduler.start()

if __name__ == "__main__":
    import socket
    local_ip = "localhost"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    log.info(f"Dashboard: http://localhost:5000")
    log.info(f"Same-network: http://{local_ip}:5000")

    # ── Cloudflare Tunnel — public URL, no account needed ─────────────────────
    import subprocess, threading, re as _re
    _cf_exe = os.path.join(os.path.dirname(__file__), "..", "cloudflared.exe")
    if os.path.exists(_cf_exe):
        def _run_tunnel():
            try:
                proc = subprocess.Popen(
                    [_cf_exe, "tunnel", "--url", "http://localhost:5000"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace"
                )
                for line in proc.stdout:
                    m = _re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
                    if m:
                        import tunnel as _tunnel
                        _tunnel.PUBLIC_URL = m.group()
                        # Write to file so emailer (same process) can read it
                        with open("public_url.txt", "w") as f:
                            f.write(m.group())
                        log.info(f"")
                        log.info(f"  PUBLIC DASHBOARD URL (any device, anywhere):")
                        log.info(f"  {_tunnel.PUBLIC_URL}")
                        log.info(f"")
                        break
            except Exception as e:
                log.warning(f"Cloudflare tunnel error: {e}")
        threading.Thread(target=_run_tunnel, daemon=True).start()
    else:
        log.info(f"Same-network access: http://{local_ip}:5000")

    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
