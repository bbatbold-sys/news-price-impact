"""
Rule-based extreme/breaking news detector.
Detects: product launches, accidents, geopolitical crises.
Returns trading recommendations instantly — no ML needed.
"""
import re, logging
log = logging.getLogger("extreme")

# ── Company name → ticker ─────────────────────────────────────────────────────
COMPANY_MAP = {
    "apple": "AAPL", "iphone": "AAPL", "ipad": "AAPL", "macbook": "AAPL",
    "airpods": "AAPL", "apple vision": "AAPL", "apple watch": "AAPL",
    "tesla": "TSLA", "elon musk": "TSLA", "cybertruck": "TSLA",
    "nvidia": "NVDA", "geforce": "NVDA",
    "amazon": "AMZN", "aws": "AMZN",
    "microsoft": "MSFT", "azure": "MSFT", "copilot": "MSFT",
    "openai": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL",
    "deepmind": "GOOGL", "gemini": "GOOGL",
    "meta": "META", "facebook": "META", "instagram": "META",
    "whatsapp": "META", "zuckerberg": "META",
    "boeing": "BA",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "coinbase": "COIN",
    "palantir": "PLTR",
    "netflix": "NFLX",
    "uber": "UBER",
    "ford": "F",
    "intel": "INTC",
    "amd": "AMD",
    "snapchat": "SNAP", "snap inc": "SNAP",
    "disney": "DIS",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman sachs": "GS", "goldman": "GS",
    "shopify": "SHOP",
    "rivian": "RIVN",
    "lucid": "LCID",
    "nio": "NIO",
    "gamestop": "GME",
    "robinhood": "HOOD",
    "abbvie": "ABBV",
    "eli lilly": "LLY", "lilly": "LLY",
    "exxon": "XOM",
    "chevron": "CVX",
}

# ── Triggers: positive event → BUY ───────────────────────────────────────────
POSITIVE_TRIGGERS = [
    "announces new", "unveils", "launches", "new product", "releases",
    "breakthrough", "beats earnings", "record revenue", "record profit",
    "fda approves", "fda approved", "approved by", "wins contract",
    "partnership", "acquires", "merger", "ipo", "all-time high",
    "raises guidance", "upgraded", "buyback", "landmark deal",
    "major deal", "blockbuster", "record sales", "surges", "soars",
]

# ── Triggers: negative event → SELL ──────────────────────────────────────────
NEGATIVE_TRIGGERS = [
    "fatal crash", "fatal accident", "car crash", "accident",
    "hits pedestrian", "kills", "killed", "death", "fatal",
    "recall", "investigation", "fraud", "lawsuit", "scandal",
    "misses earnings", "layoffs", "bankruptcy", "hacked", "data breach",
    "data leak", "fined", "banned", "collapse", "explosion",
    "fire destroys", "arrested", "charged with", "downgraded",
    "revenue miss", "profit warning", "supply chain",
]

# ── Triggers: geopolitical → BUY gold + oil ───────────────────────────────────
GEOPOLITICAL_TRIGGERS = [
    "war", "invasion", "military strike", "airstrike", "conflict",
    "sanctions", "nuclear", "troops", "coup", "terror attack",
    "missile", "escalation", "tensions between", "clash between",
    "declares war", "military operation", "ceasefire collapses",
]

GEOPOLITICAL_ASSETS = [
    ("GC=F",    "BUY", "Gold",       "Safe-haven demand surges during crises"),
    ("CL=F",    "BUY", "Crude Oil",  "Supply disruption risk"),
    ("BTC-USD", "BUY", "Bitcoin",    "Digital safe-haven demand"),
]


def detect_extreme(headline: str, description: str = "") -> list[dict]:
    """
    Returns list of extreme alerts:
    [{"asset", "action", "category", "trigger", "reason", "recommendation"}]
    Empty list = not extreme.
    """
    text  = (headline + " " + description).lower()
    alerts = []
    seen_assets = set()

    # ── 1. Positive company events → BUY ─────────────────────
    pos_hit = next((t for t in POSITIVE_TRIGGERS if t in text), None)
    if pos_hit:
        for company, ticker in COMPANY_MAP.items():
            if company in text and ticker not in seen_assets:
                seen_assets.add(ticker)
                alerts.append({
                    "asset":          ticker,
                    "action":         "BUY",
                    "category":       "product_launch",
                    "trigger":        pos_hit,
                    "reason":         f"Positive event: '{pos_hit}' about {company.title()}",
                    "recommendation": f"BUY {ticker}",
                })

    # ── 2. Negative company events → SELL ────────────────────
    neg_hit = next((t for t in NEGATIVE_TRIGGERS if t in text), None)
    if neg_hit:
        for company, ticker in COMPANY_MAP.items():
            if company in text and ticker not in seen_assets:
                seen_assets.add(ticker)
                alerts.append({
                    "asset":          ticker,
                    "action":         "SELL",
                    "category":       "negative_incident",
                    "trigger":        neg_hit,
                    "reason":         f"Negative event: '{neg_hit}' involving {company.title()}",
                    "recommendation": f"SELL {ticker}",
                })

    # ── 3. Geopolitical crisis → BUY safe havens ─────────────
    # Use word boundaries to avoid false positives (e.g. "award" contains "war")
    geo_hit = None
    for t in GEOPOLITICAL_TRIGGERS:
        pattern = r'\b' + re.escape(t) + r'\b'
        if re.search(pattern, text):
            geo_hit = t
            break
    if not geo_hit:
        if re.search(r'between \w[\w\s]{2,15} and \w[\w\s]{2,15}', text):
            geo_hit = "conflict"
    if geo_hit:
        for asset, action, name, reason in GEOPOLITICAL_ASSETS:
            if asset not in seen_assets:
                seen_assets.add(asset)
                alerts.append({
                    "asset":          asset,
                    "action":         action,
                    "category":       "geopolitical",
                    "trigger":        geo_hit,
                    "reason":         f"Geopolitical crisis ('{geo_hit}'): {reason}",
                    "recommendation": f"{action} {name} ({asset})",
                })

    if alerts:
        log.info(f"EXTREME EVENT: {len(alerts)} alert(s) | headline='{headline[:70]}'")
        for a in alerts:
            log.info(f"  → {a['recommendation']} | {a['reason']}")

    return alerts
