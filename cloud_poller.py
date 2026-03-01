"""
Cloud poller for GitHub Actions.
Polls Yahoo Finance RSS and sends email alerts for high-confidence signals.
Runs without Flask — works 24/7 even when your computer is off.
Tracks seen URLs in seen_urls.json so you never get duplicate alerts.
"""
import os, sys, json, logging

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR   = os.path.join(REPO_ROOT, "web")
SEEN_FILE = os.path.join(REPO_ROOT, "seen_urls.json")

# Change to web/ so predictor.py's relative model paths (../models_v3/) work
os.chdir(WEB_DIR)
sys.path.insert(0, WEB_DIR)
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("cloud_poller")

# ── Asset name map (same as app.py) ──────────────────────────────────────────
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


# ── Seen URL tracking ─────────────────────────────────────────────────────────
def load_seen() -> set:
    try:
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_seen(seen: set):
    # Cap at 10 000 to keep file small
    lst = sorted(seen)[-10_000:]
    with open(SEEN_FILE, "w") as f:
        json.dump(lst, f, indent=2)


# ── Write email_config.py from GitHub Secrets (env vars) ─────────────────────
def setup_email_config():
    cfg_path = os.path.join(WEB_DIR, "email_config.py")
    if os.path.exists(cfg_path):
        return   # already exists (local dev)

    recipient = os.environ.get("RECIPIENT_EMAIL", "")
    sender    = os.environ.get("SENDER_EMAIL", "")
    password  = os.environ.get("SENDER_PASSWORD", "")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    min_conf  = float(os.environ.get("MIN_CONFIDENCE", "0.62"))

    with open(cfg_path, "w") as f:
        f.write(f'RECIPIENT_EMAIL = "{recipient}"\n')
        f.write(f'SENDER_EMAIL    = "{sender}"\n')
        f.write(f'SENDER_PASSWORD = "{password}"\n')
        f.write(f'SMTP_HOST       = "{smtp_host}"\n')
        f.write(f'SMTP_PORT       = {smtp_port}\n')
        f.write(f'MIN_CONFIDENCE  = {min_conf}\n')
    log.info("email_config.py written from environment variables")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    setup_email_config()

    seen = load_seen()
    log.info(f"Loaded {len(seen)} seen URLs")

    # Load predictor — no market data, so technicals = 0, but FinBERT + ML still run
    log.info("Loading predictor (FinBERT + models)…")
    import predictor   # loads models_v3/*.pkl + FinBERT at import time
    # Note: predictor.TECHNICALS stays None — model works fine without technicals

    from poller import fetch_latest, match_assets, seconds_to_react, _fmt_react
    from emailer import send_signal_email, EMAIL_ENABLED, MIN_CONFIDENCE

    log.info("Fetching Yahoo Finance RSS…")
    articles = fetch_latest()
    new_articles = [a for a in articles if a.get("url", "") not in seen]
    log.info(f"  {len(articles)} total, {len(new_articles)} new")

    alert_signals = []

    for article in new_articles:
        url      = article.get("url", "")
        headline = article.get("title", "").strip()
        desc     = article.get("description", "")

        if not headline or not url:
            continue

        seen.add(url)

        assets = match_assets(headline)
        if not assets:
            continue

        for asset in assets[:5]:   # cap at 5 assets per headline
            try:
                result = predictor.predict(headline, asset)
                preds  = result["predictions"]
                t3     = preds["T3"]

                log.info(f"  {asset:12s} {t3['direction']:4s} "
                         f"{t3['confidence']:.0%}  \"{headline[:55]}\"")

                if (EMAIL_ENABLED
                        and t3["direction"] != "FLAT"
                        and t3["confidence"] >= MIN_CONFIDENCE):
                    alert_signals.append({
                        "asset":            asset,
                        "name":             ASSET_NAMES.get(asset, asset),
                        "asset_class":      result["asset_class"],
                        "headline":         headline,
                        "description":      desc,
                        "url":              url,
                        "direction_T1":     preds["T1"]["direction"],
                        "direction_T3":     t3["direction"],
                        "direction_T5":     preds["T5"]["direction"],
                        "pred_T1":          preds["T1"]["return_pct"],
                        "pred_T3":          t3["return_pct"],
                        "pred_T5":          preds["T5"]["return_pct"],
                        "confidence_T3":    t3["confidence"],
                        "prob_up_T3":       t3["prob_up"],
                        "prob_down_T3":     t3["prob_down"],
                        "finbert_compound": result["finbert_compound"],
                        "finbert_pos":      result["finbert_pos"],
                        "finbert_neg":      result["finbert_neg"],
                        "sentiment":        result["sentiment"],
                        "time_to_react":    _fmt_react(seconds_to_react(result["asset_class"])),
                        "detected_at":      None,
                    })
            except Exception as e:
                log.warning(f"  Predict error for {asset}: {e}")

    if alert_signals:
        log.info(f"Sending email alert: {len(alert_signals)} signal(s)…")
        try:
            send_signal_email(alert_signals)
        except Exception as e:
            log.warning(f"Email failed: {e}")
    else:
        log.info("No high-confidence signals this run.")

    save_seen(seen)
    log.info(f"Done. Tracking {len(seen)} URLs total.")


if __name__ == "__main__":
    main()
