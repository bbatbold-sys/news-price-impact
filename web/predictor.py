"""
Load v3 models once and expose a predict() function.
Also handles FinBERT scoring and feature building.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from news_price_impact_v2 import (
    compute_all_technicals, CRYPTO_ASSETS, COMMODITY_ASSETS,
    ASSET_KEYWORDS, _lookup_technicals, TECH_COLS
)
from news_price_impact_v3 import get_asset_class, V3_MODEL_DIR, extract_text_features

HORIZONS = [1, 3, 5]

# ── Load everything once at import time ──────────────────────────────────────
print("[Predictor] Loading models…")
MODELS    = {h: joblib.load(f"../{V3_MODEL_DIR}/models_T{h}.pkl") for h in HORIZONS}
FEAT_COLS = joblib.load(f"../{V3_MODEL_DIR}/feature_cols_v3.pkl")
try:
    LE_ASSET  = joblib.load(f"../{V3_MODEL_DIR}/../models/le_asset.pkl")
    LE_CLASS  = joblib.load(f"../{V3_MODEL_DIR}/../models/le_class.pkl")
except Exception:
    LE_ASSET = LE_CLASS = None

from transformers import pipeline as hf_pipeline
print("[Predictor] Loading FinBERT…")
_finbert = hf_pipeline("text-classification", model="ProsusAI/finbert",
                        device=-1, truncation=True, max_length=128, top_k=None)

PRICES      = None
TECHNICALS  = None

def load_market(market_path):
    global PRICES, TECHNICALS
    from news_price_impact_v2 import load_market_data
    PRICES     = load_market_data(market_path)
    TECHNICALS = compute_all_technicals(PRICES)
    print("[Predictor] Market data loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _finbert_score(headline: str):
    res = _finbert(headline)[0]
    s   = {r["label"]: r["score"] for r in res}
    return s.get("positive",0), s.get("negative",0), s.get("neutral",0)

def _asset_enc(asset):
    if LE_ASSET and asset in LE_ASSET.classes_:
        return int(LE_ASSET.transform([asset])[0])
    return 0

def _class_enc(cls):
    if LE_CLASS and cls in LE_CLASS.classes_:
        return int(LE_CLASS.transform([cls])[0])
    return 0

# ── Main predict function ──────────────────────────────────────────────────────
def predict(headline: str, asset: str, event_date=None) -> dict:
    """
    Returns dict with predictions for T+1, T+3, T+5.
    event_date: pd.Timestamp or None (uses latest available)
    """
    import pandas as pd
    if event_date is None:
        event_date = pd.Timestamp.now()
    elif not isinstance(event_date, pd.Timestamp):
        event_date = pd.Timestamp(event_date)

    # Sentiment
    fb_pos, fb_neg, fb_neu = _finbert_score(headline)
    fb_compound = fb_pos - fb_neg

    # Asset
    asset_class = get_asset_class(asset)
    asset_enc   = _asset_enc(asset)
    class_enc   = _class_enc(asset_class)

    # Technicals
    tech = {}
    if TECHNICALS:
        tech = _lookup_technicals(TECHNICALS, asset, event_date)

    # Text features (manual for single headline)
    BULLISH_KW = ["beats","beat","raises","upgrade","upgraded","record","surges",
                  "rallies","growth","profit","dividend","buyback","outperform",
                  "strong","boom","soars","jumps","milestone","breakthrough"]
    BEARISH_KW = ["misses","missed","cuts","downgrade","downgraded","warning",
                  "lawsuit","investigation","recall","bankrupt","layoffs",
                  "loss","decline","drops","plunges","fraud","breach"]
    tl = headline.lower()
    has_bull = int(any(k in tl for k in BULLISH_KW))
    has_bear = int(any(k in tl for k in BEARISH_KW))

    # Lag returns (use technicals as proxy if available)
    lag1 = tech.get("mom5", 0) or 0
    lag3 = tech.get("mom5", 0) or 0
    lag5 = tech.get("mom20", 0) or 0

    row = {
        "finbert_pos":      fb_pos,
        "finbert_neg":      fb_neg,
        "finbert_neu":      fb_neu,
        "finbert_compound": fb_compound,
        "day_of_week":      event_date.dayofweek,
        "month":            event_date.month,
        "is_macro":         int(any(kw in tl for kw in ASSET_KEYWORDS.get("_MACRO", []))),
        "news_count":       1,
        "asset_enc":        asset_enc,
        "asset_class_enc":  class_enc,
        "has_bullish_kw":   has_bull,
        "has_bearish_kw":   has_bear,
        "has_number":       int(any(c.isdigit() for c in headline)),
        "title_len":        len(headline),
        "kw_signal":        has_bull - has_bear,
        "lag_ret1":         lag1,
        "lag_ret3":         lag3,
        "lag_ret5":         lag5,
        "roll_sent7":       fb_compound,
    }
    for col in TECH_COLS:
        row[col] = tech.get(col, 0) or 0

    X = pd.DataFrame([row])[FEAT_COLS].fillna(0)

    label_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
    result = {
        "asset":            asset,
        "asset_class":      asset_class,
        "headline":         headline,
        "sentiment":        ("POSITIVE" if fb_compound >  0.1 else
                             "NEGATIVE" if fb_compound < -0.1 else "NEUTRAL"),
        "finbert_compound": round(fb_compound, 4),
        "finbert_pos":      round(fb_pos, 4),
        "finbert_neg":      round(fb_neg, 4),
        "predictions":      {},
    }

    for h in HORIZONS:
        from news_price_impact_v3 import blend_predict
        try:
            ret, cls_p, probs = blend_predict(X, asset, MODELS[h])
        except Exception:
            ret, cls_p, probs = 0.0, 1, [0.33,0.34,0.33]
        result["predictions"][f"T{h}"] = {
            "direction":    label_map[cls_p],
            "return_pct":   round(float(ret), 3),
            "confidence":   round(float(max(probs)), 3),
            "prob_up":      round(float(probs[2]), 3),
            "prob_flat":    round(float(probs[1]), 3),
            "prob_down":    round(float(probs[0]), 3),
        }

    return result
