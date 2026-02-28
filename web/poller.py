"""
GDELT poller — checks Yahoo Finance news every 5 minutes,
scores with FinBERT, predicts impact, stores in DB.
"""
import requests, time, logging
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd

from db import insert_signal
from predictor import predict
from news_price_impact_v2 import ASSET_KEYWORDS

log = logging.getLogger("poller")
GDELT_API   = "https://api.gdeltproject.org/api/v2/doc/doc"
POLL_WINDOW = 15   # fetch articles from last N minutes
ET_TZ       = pytz.timezone("US/Eastern")

SEEN_URLS: set = set()

# ── Market hours helper ───────────────────────────────────────────────────────
def seconds_to_react(asset_class: str) -> int:
    now_et = datetime.now(ET_TZ)
    if asset_class == "crypto":
        return 4 * 3600       # crypto never sleeps, ~4 h average

    if asset_class == "commodity":
        return 6 * 3600       # futures next session ~6 h

    # Stock — NYSE/NASDAQ 9:30-16:00 ET weekdays
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    is_weekday   = now_et.weekday() < 5

    if is_weekday and market_open <= now_et <= market_close:
        remaining = (market_close - now_et).seconds
        return min(remaining, 2 * 3600)    # react within 2 h during open

    # Find next market open
    next_open = market_open + timedelta(days=1)
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)
    return int((next_open - now_et).total_seconds())

# ── Asset matcher ─────────────────────────────────────────────────────────────
def match_assets(title: str) -> list[str]:
    tl = title.lower()
    matched, is_macro = [], False
    MACRO_ASSETS = ["NVDA","TSLA","AAPL","AMD","AMZN","META","MSFT","GOOGL",
                    "SPY","QQQ","BTC-USD","ETH-USD","GC=F","CL=F"]
    for ticker, kws in ASSET_KEYWORDS.items():
        if ticker == "_MACRO":
            if any(kw in tl for kw in kws):
                is_macro = True
            continue
        if any(kw in tl for kw in kws):
            matched.append(ticker)
    if is_macro and not matched:
        matched = MACRO_ASSETS
    elif is_macro:
        matched = list(set(matched + MACRO_ASSETS))
    return matched

# ── Fetch latest news from GDELT ──────────────────────────────────────────────
def fetch_latest():
    now  = datetime.now(timezone.utc)
    past = now - timedelta(minutes=POLL_WINDOW)
    params = {
        "query":         "domain:finance.yahoo.com",
        "mode":          "artlist",
        "maxrecords":    250,
        "startdatetime": past.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   now.strftime("%Y%m%d%H%M%S"),
        "format":        "json",
        "sort":          "DateDesc",
    }
    try:
        r = requests.get(GDELT_API, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as e:
        log.warning(f"GDELT fetch error: {e}")
        return []

# ── Process one news article ──────────────────────────────────────────────────
def process_article(article: dict):
    url      = article.get("url", "")
    headline = article.get("title", "").strip()
    seen_dt  = article.get("seendate", "")

    if not headline or not url or url in SEEN_URLS:
        return
    SEEN_URLS.add(url)

    assets = match_assets(headline)
    if not assets:
        return

    detected_at = datetime.now(timezone.utc).isoformat()
    try:
        article_date = pd.to_datetime(seen_dt[:8], format="%Y%m%d").isoformat()
    except Exception:
        article_date = detected_at

    for asset in assets:
        try:
            result      = predict(headline, asset)
            preds       = result["predictions"]
            asset_class = result["asset_class"]

            row = {
                "url":             url,
                "headline":        headline,
                "article_date":    article_date,
                "detected_at":     detected_at,
                "asset":           asset,
                "asset_class":     asset_class,
                "sentiment":       result["sentiment"],
                "finbert_compound":result["finbert_compound"],
                "direction_T1":    preds["T1"]["direction"],
                "direction_T3":    preds["T3"]["direction"],
                "direction_T5":    preds["T5"]["direction"],
                "pred_return_T1":  preds["T1"]["return_pct"],
                "pred_return_T3":  preds["T3"]["return_pct"],
                "pred_return_T5":  preds["T5"]["return_pct"],
                "confidence_T1":   preds["T1"]["confidence"],
                "confidence_T3":   preds["T3"]["confidence"],
                "confidence_T5":   preds["T5"]["confidence"],
                "prob_up_T3":      preds["T3"]["prob_up"],
                "prob_down_T3":    preds["T3"]["prob_down"],
                "seconds_to_react":seconds_to_react(asset_class),
            }
            insert_signal(row)
            log.info(f"  {asset} {preds['T3']['direction']} "
                     f"{preds['T3']['confidence']:.0%}  \"{headline[:60]}\"")
        except Exception as e:
            log.warning(f"  predict error for {asset}: {e}")

# ── Main poll loop (called by APScheduler) ────────────────────────────────────
def poll():
    log.info("Polling GDELT…")
    articles = fetch_latest()
    new = [a for a in articles if a.get("url","") not in SEEN_URLS]
    log.info(f"  {len(articles)} articles, {len(new)} new")
    for article in new:
        process_article(article)
