"""
Lightweight RSS poller — no PyTorch, no FinBERT.
Polls Yahoo Finance RSS every 60s, writes to news_live.json.
Uses keyword sentiment; optionally loads XGBoost/LightGBM models for predictions.
"""
import requests, xml.etree.ElementTree as ET, json, os, time, threading, logging
import numpy as np
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

log = logging.getLogger("light_poller")

_DIR      = os.path.dirname(os.path.abspath(__file__))
OUT_FILE  = os.path.join(_DIR, "news_live.json")
HEADERS   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

YF_GENERAL = "https://finance.yahoo.com/news/rssindex"
YF_TICKER  = "https://finance.yahoo.com/rss/headline?s={}"

TOP_TICKERS = [
    "NVDA","TSLA","AAPL","AMD","AMZN","META","MSFT","GOOGL","PLTR",
    "SPY","QQQ","JPM","GS","BAC","COIN","BTC-USD","ETH-USD","SOL-USD",
    "GC=F","CL=F","NFLX","UBER","LLY","XOM",
]

BULLISH_KW = ["beats","beat","raises","upgrade","upgraded","record","surges","rallies",
              "growth","profit","dividend","buyback","outperform","strong","soars","jumps",
              "milestone","breakthrough","raises guidance","record high","beats estimates"]
BEARISH_KW = ["misses","missed","cuts","downgrade","downgraded","warning","lawsuit",
              "investigation","recall","bankrupt","layoffs","fired","loss","decline",
              "drops","plunges","fraud","breach","shortage","below expectations","misses estimates"]

ASSET_KW = {
    "NVDA":    ["nvidia","nvda"],
    "TSLA":    ["tesla","tsla","elon musk"],
    "AAPL":    ["apple ","iphone","ipad"," mac ","apple inc"],
    "AMD":     [" amd ","advanced micro devices"],
    "AMZN":    ["amazon","aws ","amzn"],
    "META":    ["meta platforms","facebook","instagram","meta "],
    "MSFT":    ["microsoft","msft","azure","copilot"],
    "GOOGL":   ["google","alphabet","googl","gemini ai"],
    "PLTR":    ["palantir","pltr"],
    "INTC":    ["intel ","intc"],
    "COIN":    ["coinbase","coin "],
    "JPM":     ["jpmorgan","jp morgan","jamie dimon"],
    "BAC":     ["bank of america","bofa"],
    "GS":      ["goldman sachs","goldman"],
    "MS":      ["morgan stanley"],
    "V":       ["visa "],
    "MA":      ["mastercard"],
    "NFLX":    ["netflix","nflx"],
    "DIS":     ["disney ","dis "],
    "UBER":    ["uber "],
    "SNAP":    ["snap ","snapchat"],
    "SHOP":    ["shopify"],
    "LLY":     ["eli lilly"," lly "],
    "ABBV":    ["abbvie"],
    "PFE":     ["pfizer"],
    "MRNA":    ["moderna"],
    "XOM":     ["exxon","exxonmobil"],
    "CVX":     ["chevron"],
    "OXY":     ["occidental"],
    "BA":      ["boeing "],
    "GE":      ["ge aerospace","ge "],
    "SPY":     ["s&p 500","s&p500"," spy ","stock market","wall street"],
    "QQQ":     ["nasdaq","qqq"],
    "IWM":     ["russell 2000"],
    "BTC-USD": ["bitcoin","btc"],
    "ETH-USD": ["ethereum","ether","eth "],
    "SOL-USD": ["solana"," sol "],
    "XRP-USD": [" xrp ","ripple"],
    "GC=F":    ["gold price","gold futures","xau","gold market"],
    "SI=F":    ["silver price","silver futures"],
    "CL=F":    ["crude oil","oil price","wti","brent crude"],
    "NG=F":    ["natural gas"],
}

CRYPTO_ASSETS    = {"BTC-USD","ETH-USD","SOL-USD","XRP-USD","USDT-USD"}
COMMODITY_ASSETS = {"CL=F","BZ=F","NG=F","GC=F","SI=F","ZC=F","ZS=F","ZW=F","HG=F","KC=F"}

# ── Optional: load XGBoost/LightGBM models (no torch needed) ──────────────────
_models     = {}
_feat_cols  = []
_models_ok  = False

def _try_load_models():
    global _models, _feat_cols, _models_ok
    try:
        import joblib
        root = os.path.join(_DIR, "..")
        feat_path = os.path.join(root, "models_v3", "feature_cols_v3.pkl")
        _feat_cols = joblib.load(feat_path)
        for h in [1, 3, 5]:
            _models[h] = joblib.load(os.path.join(root, "models_v3", f"models_T{h}.pkl"))
        _models_ok = True
        log.info(f"Models loaded: T+1, T+3, T+5  ({len(_feat_cols)} features)")
    except Exception as e:
        log.warning(f"Models not loaded (predictions disabled): {e}")

threading.Thread(target=_try_load_models, daemon=True).start()


def asset_class(asset):
    if asset in CRYPTO_ASSETS:    return "crypto"
    if asset in COMMODITY_ASSETS: return "commodity"
    return "stock"


def simple_sentiment(title):
    tl = title.lower()
    bull = sum(1 for k in BULLISH_KW if k in tl)
    bear = sum(1 for k in BEARISH_KW if k in tl)
    if bull > bear:   return "POSITIVE",  0.6,  0.1, 0.3
    if bear > bull:   return "NEGATIVE",  0.1,  0.6, 0.3
    return "NEUTRAL", 0.2, 0.2, 0.6


def match_assets(title):
    tl = title.lower()
    MACRO_KW = ["federal reserve","fed rate","interest rate","inflation","recession",
                "gdp","earnings season","market rally","market selloff","tariff"]
    MACRO_ASSETS = ["NVDA","TSLA","AAPL","AMD","AMZN","META","MSFT","GOOGL",
                    "SPY","QQQ","BTC-USD","ETH-USD","GC=F","CL=F"]
    matched, is_macro = [], False
    if any(k in tl for k in MACRO_KW): is_macro = True
    for ticker, kws in ASSET_KW.items():
        if any(k in tl for k in kws): matched.append(ticker)
    if is_macro and not matched: return MACRO_ASSETS
    if is_macro: return list(set(matched + MACRO_ASSETS))
    return matched


def _predict_simple(headline, asset, sent_compound):
    """Best-effort prediction using XGBoost/LightGBM without FinBERT features."""
    if not _models_ok or not _feat_cols:
        return None
    try:
        import pandas as pd
        row = {f: 0.0 for f in _feat_cols}
        # FinBERT proxies from keyword sentiment
        row["finbert_compound"] = sent_compound
        if "finbert_pos" in row: row["finbert_pos"] = max(0, sent_compound)
        if "finbert_neg" in row: row["finbert_neg"] = max(0, -sent_compound)
        if "finbert_neu" in row: row["finbert_neu"] = 1 - abs(sent_compound)
        # Text keyword features
        tl = headline.lower()
        if "has_bullish_kw" in row:
            row["has_bullish_kw"] = int(any(k in tl for k in BULLISH_KW))
        if "has_bearish_kw" in row:
            row["has_bearish_kw"] = int(any(k in tl for k in BEARISH_KW))
        if "title_len" in row:
            row["title_len"] = float(len(headline))
        if "kw_signal" in row:
            row["kw_signal"] = float(row.get("has_bullish_kw",0) - row.get("has_bearish_kw",0))

        X = pd.DataFrame([row])[_feat_cols].fillna(0)
        cls = asset_class(asset)
        preds = {}
        LABEL = {0:"DOWN", 1:"FLAT", 2:"UP"}
        for h in [1, 3, 5]:
            m = _models.get(h, {})
            if cls not in m: m_cls = m.get("stock", {})
            else:             m_cls = m[cls]
            if not m_cls: continue
            alpha = 0.5
            ret = alpha * m_cls["xgb_reg"].predict(X)[0] + (1-alpha) * m_cls["lgb_reg"].predict(X)[0]
            pb_xgb = m_cls["xgb_clf"].predict_proba(X)[0]
            pb_lgb = m_cls["lgb_clf"].predict_proba(X)[0]
            pb = alpha * pb_xgb + (1-alpha) * pb_lgb
            ci = int(np.argmax(pb))
            preds[f"T{h}"] = {
                "direction":   LABEL[ci],
                "return_pct":  round(float(ret), 4),
                "confidence":  round(float(pb[ci]), 4),
                "prob_up":     round(float(pb[2]), 4),
                "prob_down":   round(float(pb[0]), 4),
            }
        return preds if preds else None
    except Exception as e:
        log.debug(f"Predict error ({asset}): {e}")
        return None


# ── RSS fetch ──────────────────────────────────────────────────────────────────
def _fetch_rss(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        for item in root.findall(".//item"):
            t = item.findtext("title","").strip()
            l = item.findtext("link","").strip()
            p = item.findtext("pubDate","")
            d = item.findtext("description","").strip()
            if t and l: out.append({"title":t,"url":l,"pubDate":p,"desc":d})
        return out
    except Exception as e:
        log.debug(f"RSS error {url}: {e}")
        return []


# ── Shared state ───────────────────────────────────────────────────────────────
_seen      : set  = set()
_live_news : list = []
_lock               = threading.Lock()


def _load_existing():
    """Pre-load seen URLs and existing news on startup."""
    global _live_news
    try:
        with open(OUT_FILE) as f:
            _live_news = json.load(f)
            for n in _live_news:
                if n.get("url"): _seen.add(n["url"])
        log.info(f"Loaded {len(_live_news)} existing articles from disk")
    except Exception:
        pass


def poll_once():
    global _live_news
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_articles = {}
    urls = [YF_GENERAL] + [YF_TICKER.format(t) for t in TOP_TICKERS]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_rss, u): u for u in urls}
        for fut in as_completed(futures):
            for a in fut.result():
                all_articles[a["url"]] = a

    now     = datetime.now(timezone.utc).isoformat()
    new_rows = []

    for art in all_articles.values():
        url = art["url"]
        if url in _seen: continue
        _seen.add(url)

        title  = art["title"]
        assets = match_assets(title)
        if not assets: continue

        try: pub = parsedate_to_datetime(art["pubDate"]).isoformat()
        except: pub = now

        sent_label, fb_pos, fb_neg, fb_neu = simple_sentiment(title)
        sent_compound = fb_pos - fb_neg

        for asset in assets[:4]:
            cls = asset_class(asset)
            preds = _predict_simple(title, asset, sent_compound) if _models_ok else None

            row = {
                "headline":         title,
                "url":              url,
                "asset":            asset,
                "asset_class":      cls,
                "sentiment":        sent_label,
                "finbert_compound": round(sent_compound, 3),
                "finbert_pos":      round(fb_pos, 3),
                "finbert_neg":      round(fb_neg, 3),
                "detected_at":      now,
                "article_date":     pub,
                "direction_T1":     preds["T1"]["direction"]   if preds else None,
                "direction_T3":     preds["T3"]["direction"]   if preds else None,
                "direction_T5":     preds["T5"]["direction"]   if preds else None,
                "pred_return_T1":   preds["T1"]["return_pct"]  if preds else None,
                "pred_return_T3":   preds["T3"]["return_pct"]  if preds else None,
                "pred_return_T5":   preds["T5"]["return_pct"]  if preds else None,
                "confidence_T3":    preds["T3"]["confidence"]  if preds else None,
                "prob_up_T3":       preds["T3"]["prob_up"]     if preds else None,
                "prob_down_T3":     preds["T3"]["prob_down"]   if preds else None,
                "seconds_to_react": None,
            }
            new_rows.append(row)

    if new_rows:
        with _lock:
            _live_news = (new_rows + _live_news)[:500]
        try:
            with open(OUT_FILE, "w") as f:
                json.dump(_live_news, f)
        except Exception as e:
            log.warning(f"Write error: {e}")
        log.info(f"Poller: +{len(new_rows)} new signals ({len(new_rows)//max(1,len(set(r['url'] for r in new_rows)))} assets/article avg)")


def get_live_news():
    with _lock:
        return list(_live_news)


def start():
    _load_existing()
    def _loop():
        try:    poll_once()
        except Exception as e: log.warning(f"First poll error: {e}")
        while True:
            time.sleep(60)
            try:    poll_once()
            except Exception as e: log.warning(f"Poll loop error: {e}")
    threading.Thread(target=_loop, daemon=True).start()
    log.info("Light poller started — polling every 60s")
