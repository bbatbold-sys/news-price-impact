"""
Sends rich HTML email alerts with full article content and highlighted keywords.
"""
import smtplib, logging, requests, re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
from bs4 import BeautifulSoup

log = logging.getLogger("emailer")

try:
    from email_config import (RECIPIENT_EMAIL, SENDER_EMAIL, SENDER_PASSWORD,
                               SMTP_HOST, SMTP_PORT, MIN_CONFIDENCE)
    EMAIL_ENABLED = (
        SENDER_PASSWORD != "xxxx xxxx xxxx xxxx" and
        RECIPIENT_EMAIL != "YOUR_EMAIL@gmail.com"
    )
except ImportError:
    EMAIL_ENABLED = False
    MIN_CONFIDENCE = 0.62

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

BULLISH_KW = ["beats","beat","raises","upgrade","upgraded","record","surges","rallies",
              "growth","profit","dividend","buyback","outperform","strong","boom",
              "soars","jumps","milestone","breakthrough","exceeds","record-high",
              "all-time high","bullish","buy","overweight","positive"]

BEARISH_KW = ["misses","missed","cuts","downgrade","downgraded","warning","lawsuit",
              "investigation","recall","bankrupt","layoffs","loss","decline","drops",
              "plunges","fraud","breach","disappoints","disappointing","bearish",
              "sell","underweight","negative","concern","risk","crash","collapse"]


# ── Scrape article body from URL ──────────────────────────────────────────────
def scrape_article(url: str, max_chars: int = 1200) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        soup = BeautifulSoup(r.content, "html.parser")

        # Remove script/style noise
        for tag in soup(["script","style","nav","header","footer","aside","form"]):
            tag.decompose()

        # Try known article containers in order
        body = (soup.find("div", {"class": "caas-body"}) or
                soup.find("div", {"class": "article-body"}) or
                soup.find("div", {"itemprop": "articleBody"}) or
                soup.find("article") or
                soup.find("main"))

        if body:
            paragraphs = [p.get_text(" ", strip=True)
                          for p in body.find_all("p")
                          if len(p.get_text(strip=True)) > 40]
            text = " ".join(paragraphs[:6])
            return text[:max_chars] + ("…" if len(text) > max_chars else "")
    except Exception as e:
        log.debug(f"Article scrape failed ({url}): {e}")
    return ""


# ── Highlight bullish/bearish keywords in HTML text ───────────────────────────
def highlight(text: str) -> str:
    if not text:
        return ""
    # Escape HTML first
    text = (text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    for kw in BULLISH_KW:
        pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
        text = pattern.sub(
            r'<mark style="background:#166534;color:#bbf7d0;padding:1px 4px;'
            r'border-radius:3px;font-weight:bold">\1</mark>', text)
    for kw in BEARISH_KW:
        pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
        text = pattern.sub(
            r'<mark style="background:#7f1d1d;color:#fecaca;padding:1px 4px;'
            r'border-radius:3px;font-weight:bold">\1</mark>', text)
    return text


def _signal_color(direction: str) -> str:
    return {"UP": "#22c55e", "DOWN": "#ef4444", "FLAT": "#94a3b8"}.get(direction, "#94a3b8")

def _signal_bg(direction: str) -> str:
    return {"UP": "#14532d", "DOWN": "#450a0a", "FLAT": "#1e293b"}.get(direction, "#1e293b")

def _bar(pct: float, color: str) -> str:
    w = int(pct * 100)
    return (f'<div style="background:#334155;border-radius:4px;height:8px;width:120px;display:inline-block">'
            f'<div style="background:{color};height:8px;border-radius:4px;width:{w}px"></div></div>')


# ── Build one signal card ─────────────────────────────────────────────────────
def _signal_card(s: dict, article_text: str) -> str:
    color   = _signal_color(s["direction_T3"])
    bg      = _signal_bg(s["direction_T3"])
    conf    = s.get("confidence_T3", 0)
    ret_str = f"{s['pred_T3']:+.2f}%" if s.get("pred_T3") is not None else "N/A"
    conf_pct = f"{conf*100:.0f}%"
    hl_headline = highlight(s.get("headline",""))
    hl_desc     = highlight(s.get("description",""))
    hl_body     = highlight(article_text) if article_text else ""

    pos_pct = int(s.get("finbert_pos", 0) * 100)
    neg_pct = int(s.get("finbert_neg", 0) * 100)

    return f"""
    <div style="background:{bg};border:1px solid {color};border-radius:12px;
                padding:20px;margin-bottom:24px">

      <!-- Header row -->
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:14px;flex-wrap:wrap;gap:8px">
        <div>
          <span style="font-size:22px;font-weight:800;color:#f1f5f9">
            {s.get('name', s['asset'])}
          </span>
          <span style="color:#64748b;font-size:13px;margin-left:8px">
            ({s['asset']} &bull; {s['asset_class'].upper()})
          </span>
        </div>
        <div style="text-align:right">
          <span style="font-size:28px;font-weight:900;color:{color}">
            {s["direction_T3"]}
          </span>
          <span style="color:{color};font-size:14px;margin-left:6px">{ret_str}</span>
        </div>
      </div>

      <!-- Stats row -->
      <table style="width:100%;margin-bottom:16px">
        <tr>
          <td style="padding:4px 8px 4px 0;color:#94a3b8;font-size:12px;
                     text-transform:uppercase;width:140px">Confidence</td>
          <td style="padding:4px 0">
            {_bar(conf, color)}
            <span style="color:{color};font-weight:bold;margin-left:8px">{conf_pct}</span>
          </td>
          <td style="padding:4px 8px 4px 20px;color:#94a3b8;font-size:12px;
                     text-transform:uppercase;width:120px">Sentiment</td>
          <td style="padding:4px 0;color:#f1f5f9">{s.get('sentiment','')}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px 4px 0;color:#94a3b8;font-size:12px;
                     text-transform:uppercase">React In</td>
          <td style="padding:4px 0;color:#f59e0b;font-weight:bold">
            {s.get('time_to_react','')}
          </td>
          <td style="padding:4px 8px 4px 20px;color:#94a3b8;font-size:12px;
                     text-transform:uppercase">FinBERT</td>
          <td style="padding:4px 0;font-size:12px">
            <span style="color:#22c55e">+{pos_pct}%</span>
            &nbsp;&nbsp;
            <span style="color:#ef4444">-{neg_pct}%</span>
          </td>
        </tr>
      </table>

      <!-- Headline -->
      <div style="background:#0f172a;border-radius:8px;padding:12px;margin-bottom:12px">
        <div style="color:#64748b;font-size:11px;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px">Headline</div>
        <div style="color:#e2e8f0;font-size:15px;font-weight:600;line-height:1.5">
          {hl_headline}
        </div>
      </div>

      <!-- Summary from RSS -->
      {f'''<div style="background:#0f172a;border-radius:8px;padding:12px;margin-bottom:12px">
        <div style="color:#64748b;font-size:11px;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px">Summary</div>
        <div style="color:#cbd5e1;font-size:14px;line-height:1.6">{hl_desc}</div>
      </div>''' if hl_desc else ''}

      <!-- Article body -->
      {f'''<div style="background:#0f172a;border-radius:8px;padding:12px;margin-bottom:12px">
        <div style="color:#64748b;font-size:11px;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px">Article Content</div>
        <div style="color:#cbd5e1;font-size:13px;line-height:1.7">{hl_body}</div>
      </div>''' if hl_body else ''}

      <!-- Legend + link -->
      <div style="display:flex;justify-content:space-between;align-items:center;
                  flex-wrap:wrap;gap:8px">
        <div style="font-size:11px;color:#475569">
          <mark style="background:#166534;color:#bbf7d0;padding:1px 4px;
                       border-radius:3px">green</mark> = bullish keyword &nbsp;
          <mark style="background:#7f1d1d;color:#fecaca;padding:1px 4px;
                       border-radius:3px">red</mark> = bearish keyword
        </div>
        <a href="{s.get('url','#')}"
           style="color:#38bdf8;font-size:13px;text-decoration:none">
          Read full article &rarr;
        </a>
      </div>
    </div>"""


# ── Main send function ────────────────────────────────────────────────────────
def send_signal_email(signals: list[dict]):
    if not EMAIL_ENABLED or not signals:
        return

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    count   = len(signals)
    subject = (f"[{'UP' if signals[0]['direction_T3']=='UP' else 'DOWN'} Signal] "
               f"{signals[0].get('name', signals[0]['asset'])} "
               f"{signals[0]['confidence_T3']*100:.0f}% conf"
               + (f" + {count-1} more" if count > 1 else ""))

    # Scrape article bodies
    cards_html = ""
    for s in signals:
        article_text = scrape_article(s.get("url", ""))
        cards_html  += _signal_card(s, article_text)

    try:
        import tunnel as _tunnel
        dashboard_url = _tunnel.PUBLIC_URL
        # Also check file (updated by tunnel thread)
        import os as _os
        _url_file = _os.path.join(_os.path.dirname(__file__), "public_url.txt")
        if _os.path.exists(_url_file):
            with open(_url_file) as f:
                _url = f.read().strip()
            if _url:
                dashboard_url = _url
                _tunnel.PUBLIC_URL = _url
    except Exception:
        dashboard_url = "http://localhost:5000"

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;
                       margin:0;padding:24px">
      <div style="max-width:720px;margin:auto">

        <!-- Title bar -->
        <div style="background:#1e293b;border-radius:12px;padding:20px;
                    margin-bottom:24px;border-left:4px solid #38bdf8">
          <h2 style="color:#38bdf8;margin:0 0 4px 0;font-size:22px">
            News Impact Alert
          </h2>
          <p style="color:#64748b;margin:0;font-size:13px">
            {now_str} &bull; {count} high-confidence signal{'s' if count > 1 else ''}
            &bull; <a href="{dashboard_url}" style="color:#38bdf8">Open Dashboard</a>
          </p>
        </div>

        {cards_html}

        <p style="color:#334155;font-size:11px;text-align:center;margin-top:8px">
          Min confidence: {MIN_CONFIDENCE*100:.0f}% &bull;
          Dashboard: <a href="{dashboard_url}" style="color:#475569">{dashboard_url}</a>
        </p>
      </div>
    </body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        log.info(f"Email sent: {count} signal(s) to {RECIPIENT_EMAIL}")
    except Exception as e:
        log.warning(f"Email send failed: {e}")
