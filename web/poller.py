"""
Yahoo Finance RSS poller — fetches news directly from Yahoo Finance
with zero delay. Polls every 1 minute.
"""
import requests, logging, xml.etree.ElementTree as ET, json, os, subprocess, time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import pytz
import pandas as pd

from db import insert_signal
from predictor import predict
from news_price_impact_v2 import ASSET_KEYWORDS

log     = logging.getLogger("poller")
ET_TZ   = pytz.timezone("US/Eastern")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── Shared-state files (in repo root, synced to GitHub so Actions can read them)
_REPO_ROOT     = os.path.join(os.path.dirname(__file__), "..")
_SEEN_FILE     = os.path.join(_REPO_ROOT, "seen_urls.json")
_HEARTBEAT_FILE= os.path.join(_REPO_ROOT, "heartbeat.txt")
_poll_count    = 0        # counts polls; we push to git every 5 runs
_last_push     = 0.0     # time.time() of last git push

SEEN_URLS: set = set()

def _load_seen():
    """Load seen URLs from file into memory (called once at startup)."""
    global SEEN_URLS
    try:
        with open(_SEEN_FILE) as f:
            SEEN_URLS = set(json.load(f))
        log.info(f"Loaded {len(SEEN_URLS)} seen URLs from disk")
    except Exception:
        pass

def _sync_to_github():
    """Write heartbeat + seen URLs, then git push (non-blocking, best-effort)."""
    global _last_push
    try:
        # Write heartbeat timestamp
        with open(_HEARTBEAT_FILE, "w") as f:
            f.write(str(int(time.time())))

        # Write seen URLs (cap at 10 000)
        with open(_SEEN_FILE, "w") as f:
            json.dump(sorted(SEEN_URLS)[-10_000:], f)

        # Git push in background — won't block the poller
        def _push():
            try:
                repo = os.path.abspath(_REPO_ROOT)
                subprocess.run(["git", "add", "seen_urls.json", "heartbeat.txt"],
                               cwd=repo, capture_output=True, timeout=15)
                res = subprocess.run(
                    ["git", "commit", "-m", "chore: local poller sync [skip ci]"],
                    cwd=repo, capture_output=True, timeout=15)
                if res.returncode == 0:   # only push if there was something new
                    subprocess.run(["git", "push", "origin", "master"],
                                   cwd=repo, capture_output=True, timeout=30)
            except Exception as e:
                log.debug(f"Git sync error: {e}")

        import threading
        threading.Thread(target=_push, daemon=True).start()
        _last_push = time.time()
    except Exception as e:
        log.debug(f"Sync error: {e}")

# Load persisted seen URLs at import time
_load_seen()

# ── Top tickers to poll individually (most active / news-heavy) ───────────────
TOP_TICKERS = [
    "NVDA","TSLA","AAPL","AMD","AMZN","META","MSFT","GOOGL","PLTR",
    "SPY","QQQ","JPM","GS","BAC","COIN",
    "BTC-USD","ETH-USD","SOL-USD",
    "GC=F","CL=F","NG=F",
]

YF_GENERAL_RSS = "https://finance.yahoo.com/news/rssindex"
YF_TICKER_RSS  = "https://finance.yahoo.com/rss/headline?s={ticker}"


def _fmt_react(s):
    if s is None: return "Unknown"
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    h = s // 3600; m = (s % 3600) // 60
    return f"{h}h {m}m" if m else f"{h}h"


# ── Market hours helper ───────────────────────────────────────────────────────
def seconds_to_react(asset_class: str) -> int:
    now_et = datetime.now(ET_TZ)
    if asset_class == "crypto":
        return 4 * 3600
    if asset_class == "commodity":
        return 6 * 3600

    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    is_weekday   = now_et.weekday() < 5

    if is_weekday and market_open <= now_et <= market_close:
        remaining = (market_close - now_et).seconds
        return min(remaining, 2 * 3600)

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


# ── Fetch one RSS feed and return list of articles ────────────────────────────
def _fetch_rss(url: str) -> list[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        articles = []
        for item in root.findall(".//item"):
            title    = item.findtext("title", "").strip()
            link     = item.findtext("link",  "").strip()
            pub_date = item.findtext("pubDate", "")
            if title and link:
                desc = item.findtext("description", "").strip()
                articles.append({"title": title, "url": link, "pubDate": pub_date, "description": desc})
        return articles
    except Exception as e:
        log.warning(f"RSS fetch error ({url}): {e}")
        return []


# ── Fetch all feeds (general + top tickers) ───────────────────────────────────
def fetch_latest() -> list[dict]:
    all_articles = {}   # url -> article (deduplicate)

    # General feed
    for a in _fetch_rss(YF_GENERAL_RSS):
        all_articles[a["url"]] = a

    # Per-ticker feeds
    for ticker in TOP_TICKERS:
        for a in _fetch_rss(YF_TICKER_RSS.format(ticker=ticker)):
            all_articles[a["url"]] = a

    return list(all_articles.values())


# ── Process one news article ──────────────────────────────────────────────────
def process_article(article: dict):
    url         = article.get("url", "")
    headline    = article.get("title", "").strip()
    pub_date    = article.get("pubDate", "")
    description = article.get("description", "")

    if not headline or not url or url in SEEN_URLS:
        return
    SEEN_URLS.add(url)

    assets = match_assets(headline)
    if not assets:
        return

    detected_at = datetime.now(timezone.utc).isoformat()
    try:
        article_date = parsedate_to_datetime(pub_date).isoformat()
    except Exception:
        article_date = detected_at

    alert_signals = []
    for asset in assets:
        try:
            result      = predict(headline, asset)
            preds       = result["predictions"]
            asset_class = result["asset_class"]

            row = {
                "url":              url,
                "headline":         headline,
                "article_date":     article_date,
                "detected_at":      detected_at,
                "asset":            asset,
                "asset_class":      asset_class,
                "sentiment":        result["sentiment"],
                "finbert_compound": result["finbert_compound"],
                "direction_T1":     preds["T1"]["direction"],
                "direction_T3":     preds["T3"]["direction"],
                "direction_T5":     preds["T5"]["direction"],
                "pred_return_T1":   preds["T1"]["return_pct"],
                "pred_return_T3":   preds["T3"]["return_pct"],
                "pred_return_T5":   preds["T5"]["return_pct"],
                "confidence_T1":    preds["T1"]["confidence"],
                "confidence_T3":    preds["T3"]["confidence"],
                "confidence_T5":    preds["T5"]["confidence"],
                "prob_up_T3":       preds["T3"]["prob_up"],
                "prob_down_T3":     preds["T3"]["prob_down"],
                "seconds_to_react": seconds_to_react(asset_class),
            }
            insert_signal(row)
            log.info(f"  {asset} {preds['T3']['direction']} "
                     f"{preds['T3']['confidence']:.0%}  \"{headline[:60]}\"")

            try:
                from emailer import EMAIL_ENABLED, MIN_CONFIDENCE
                if (EMAIL_ENABLED
                        and preds["T3"]["direction"] != "FLAT"
                        and preds["T3"]["confidence"] >= MIN_CONFIDENCE):
                    from app import ASSET_NAMES
                    alert_signals.append({
                        "asset":         asset,
                        "name":          ASSET_NAMES.get(asset, asset),
                        "asset_class":   asset_class,
                        "headline":      headline,
                        "description":   description,
                        "url":           url,
                        "direction_T3":  preds["T3"]["direction"],
                        "confidence_T3": preds["T3"]["confidence"],
                        "pred_T3":       preds["T3"]["return_pct"],
                        "sentiment":     result["sentiment"],
                        "finbert_pos":   result["finbert_pos"],
                        "finbert_neg":   result["finbert_neg"],
                        "time_to_react": _fmt_react(seconds_to_react(asset_class)),
                    })
            except Exception:
                pass
        except Exception as e:
            log.warning(f"  predict error for {asset}: {e}")

    if alert_signals:
        try:
            from emailer import send_signal_email
            send_signal_email(alert_signals)
        except Exception as e:
            log.warning(f"Email alert error: {e}")


# ── Main poll loop (called by APScheduler every 1 minute) ─────────────────────
def poll():
    global _poll_count, _last_push
    log.info("Polling Yahoo Finance RSS...")
    articles = fetch_latest()
    new = [a for a in articles if a.get("url", "") not in SEEN_URLS]
    log.info(f"  {len(articles)} articles fetched, {len(new)} new")
    for article in new:
        process_article(article)

    _poll_count += 1
    # Sync to GitHub every 5 polls (~5 min) or if it's been > 4 min since last push
    if _poll_count % 5 == 0 or time.time() - _last_push > 240:
        _sync_to_github()
