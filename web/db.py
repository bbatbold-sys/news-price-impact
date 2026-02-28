"""SQLite database layer for the news signal dashboard."""
import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), "signals.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE,
    headline        TEXT,
    article_date    TEXT,
    detected_at     TEXT,
    asset           TEXT,
    asset_class     TEXT,
    sentiment       TEXT,
    finbert_compound REAL,
    direction_T1    TEXT,
    direction_T3    TEXT,
    direction_T5    TEXT,
    pred_return_T1  REAL,
    pred_return_T3  REAL,
    pred_return_T5  REAL,
    confidence_T1   REAL,
    confidence_T3   REAL,
    confidence_T5   REAL,
    prob_up_T3      REAL,
    prob_down_T3    REAL,
    seconds_to_react INTEGER
);
CREATE INDEX IF NOT EXISTS idx_asset     ON signals(asset);
CREATE INDEX IF NOT EXISTS idx_detected  ON signals(detected_at);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)

def insert_signal(row: dict):
    sql = """
    INSERT OR IGNORE INTO signals
      (url, headline, article_date, detected_at, asset, asset_class,
       sentiment, finbert_compound,
       direction_T1, direction_T3, direction_T5,
       pred_return_T1, pred_return_T3, pred_return_T5,
       confidence_T1, confidence_T3, confidence_T5,
       prob_up_T3, prob_down_T3, seconds_to_react)
    VALUES
      (:url, :headline, :article_date, :detected_at, :asset, :asset_class,
       :sentiment, :finbert_compound,
       :direction_T1, :direction_T3, :direction_T5,
       :pred_return_T1, :pred_return_T3, :pred_return_T5,
       :confidence_T1, :confidence_T3, :confidence_T5,
       :prob_up_T3, :prob_down_T3, :seconds_to_react)
    """
    with get_conn() as conn:
        conn.execute(sql, row)

def latest_per_asset():
    """Latest signal for every asset."""
    sql = """
    SELECT s.*
    FROM signals s
    INNER JOIN (
        SELECT asset, MAX(detected_at) AS mx
        FROM signals GROUP BY asset
    ) t ON s.asset = t.asset AND s.detected_at = t.mx
    ORDER BY s.asset_class, s.asset
    """
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]

def recent_news(limit=50):
    sql = """
    SELECT * FROM signals
    ORDER BY detected_at DESC LIMIT ?
    """
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]
