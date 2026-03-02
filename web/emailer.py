"""
Sends rich HTML email alerts matching the PULSE dashboard dark-space theme.
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

# ── Colour palette (matches dashboard CSS variables) ──────────────────────────
BG       = "#05070f"
BG2      = "#080c17"
SURFACE  = "#0c1222"
SURFACE2 = "#101828"
BORDER   = "#1a2540"
TEXT     = "#e8edf5"
MUTED    = "#5a6a84"
MUTED2   = "#8898b3"
CYAN     = "#00b4d8"
UP       = "#00d084"
DOWN     = "#ff4560"
GOLD     = "#fbbf24"
PURPLE   = "#7c3aed"

BULLISH_KW = ["beats","beat","raises","upgrade","upgraded","record","surges","rallies",
              "growth","profit","dividend","buyback","outperform","strong","boom",
              "soars","jumps","milestone","breakthrough","exceeds","record-high",
              "all-time high","bullish","buy","overweight","positive"]

BEARISH_KW = ["misses","missed","cuts","downgrade","downgraded","warning","lawsuit",
              "investigation","recall","bankrupt","layoffs","loss","decline","drops",
              "plunges","fraud","breach","disappoints","disappointing","bearish",
              "sell","underweight","negative","concern","risk","crash","collapse"]


# ── Scrape article body ────────────────────────────────────────────────────────
def scrape_article(url: str, max_chars: int = 1200) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script","style","nav","header","footer","aside","form"]):
            tag.decompose()
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


# ── Highlight bullish/bearish keywords ────────────────────────────────────────
def highlight(text: str) -> str:
    if not text:
        return ""
    text = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    for kw in BULLISH_KW:
        text = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE).sub(
            r'<mark style="background:#14532d;color:#86efac;padding:1px 4px;'
            r'border-radius:3px;font-weight:700">\1</mark>', text)
    for kw in BEARISH_KW:
        text = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE).sub(
            r'<mark style="background:#7f1d1d;color:#fca5a5;padding:1px 4px;'
            r'border-radius:3px;font-weight:700">\1</mark>', text)
    return text


# ── Confidence bar (180 px wide) ───────────────────────────────────────────────
def _conf_bar(conf: float, color: str) -> str:
    w = int(conf * 180)
    return (f'<div style="background:{SURFACE};border-radius:4px;height:5px;width:180px;margin:5px 0 3px">'
            f'<div style="background:{color};height:5px;border-radius:4px;width:{w}px"></div></div>')


# ── Single signal card ────────────────────────────────────────────────────────
def _signal_card(s: dict, article_text: str) -> str:
    dir3  = s["direction_T3"]
    acc   = UP if dir3 == "UP" else DOWN if dir3 == "DOWN" else MUTED
    arrow = "↑" if dir3 == "UP" else "↓" if dir3 == "DOWN" else "→"
    conf  = s.get("confidence_T3", 0)
    conf_pct = f"{conf*100:.0f}%"

    def fmt_ret(k):
        v = s.get(k)
        return (f'<span style="color:{acc};font-family:monospace;font-weight:700">'
                f'{v:+.2f}%</span>') if v is not None else '<span style="color:#3d4f6b">—</span>'

    hl_headline = highlight(s.get("headline", ""))
    hl_desc     = highlight(s.get("description", ""))
    hl_body     = highlight(article_text) if article_text else ""

    pos_pct = int(s.get("finbert_pos", 0) * 100)
    neg_pct = int(s.get("finbert_neg", 0) * 100)

    desc_block = (
        f'<tr><td colspan="2" style="padding:14px 20px;background:{BG2};'
        f'border-top:1px solid {BORDER}">'
        f'<div style="color:{MUTED};font-size:9px;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">Summary</div>'
        f'<div style="color:{MUTED2};font-size:13px;line-height:1.6">{hl_desc}</div>'
        f'</td></tr>' if hl_desc else ""
    )
    body_block = (
        f'<tr><td colspan="2" style="padding:14px 20px;background:{BG2};'
        f'border-top:1px solid {BORDER}">'
        f'<div style="color:{MUTED};font-size:9px;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">Article Content</div>'
        f'<div style="color:{MUTED2};font-size:12px;line-height:1.7">{hl_body}</div>'
        f'</td></tr>' if hl_body else ""
    )

    return f"""
<table width="100%" cellpadding="0" cellspacing="0"
       style="background:{SURFACE};border:1px solid {acc};border-left:4px solid {acc};
              border-radius:12px;margin-bottom:20px;overflow:hidden;border-collapse:separate">
  <!-- Header -->
  <tr>
    <td colspan="2"
        style="background:linear-gradient(135deg,{SURFACE},{acc}18);
               padding:18px 20px;border-bottom:1px solid {BORDER}">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="vertical-align:top">
            <div style="font-size:21px;font-weight:900;color:{TEXT};letter-spacing:-.3px">
              {s.get('name', s['asset'])}
            </div>
            <div style="font-size:11px;color:{MUTED};margin-top:2px">
              {s['asset']} &bull; {s['asset_class'].upper()}
            </div>
          </td>
          <td style="text-align:right;vertical-align:top">
            <div style="font-size:30px;font-weight:900;color:{acc};line-height:1;font-family:monospace">
              {arrow} {dir3}
            </div>
            <div style="font-size:16px;font-weight:700;color:{acc};font-family:monospace;margin-top:2px">
              {fmt_ret('pred_T3')}
            </div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Stats -->
  <tr>
    <td style="padding:16px 20px;vertical-align:top;width:55%">
      <div style="color:{MUTED};font-size:9px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
        Confidence
      </div>
      {_conf_bar(conf, acc)}
      <div style="font-size:22px;font-weight:900;color:{acc};font-family:monospace">{conf_pct}</div>
      <div style="margin-top:12px">
        <span style="background:{SURFACE2};color:{MUTED2};padding:3px 9px;border-radius:6px;
                     font-size:11px;border:1px solid {BORDER}">{s.get('sentiment','—')}</span>
        &nbsp;
        <span style="color:{UP};font-size:11px;font-family:monospace">+{pos_pct}%</span>
        <span style="color:{DOWN};font-size:11px;font-family:monospace"> -{neg_pct}%</span>
      </div>
      <div style="margin-top:10px;color:{GOLD};font-size:12px;font-weight:700;font-family:monospace">
        ⏱ {s.get('time_to_react','—')}
        <span style="color:{MUTED};font-weight:400;font-size:10px"> to react</span>
      </div>
    </td>
    <td style="padding:16px 20px;vertical-align:top;border-left:1px solid {BORDER}">
      <table cellpadding="5" cellspacing="0" width="100%">
        <tr>
          <td style="color:{MUTED};font-size:9px;text-transform:uppercase;letter-spacing:1px">T+1 Day</td>
          <td style="text-align:right">{fmt_ret('pred_T1')}</td>
        </tr>
        <tr>
          <td style="color:{MUTED};font-size:9px;text-transform:uppercase;letter-spacing:1px">T+3 Days</td>
          <td style="text-align:right">{fmt_ret('pred_T3')}</td>
        </tr>
        <tr>
          <td style="color:{MUTED};font-size:9px;text-transform:uppercase;letter-spacing:1px">T+5 Days</td>
          <td style="text-align:right">{fmt_ret('pred_T5')}</td>
        </tr>
        <tr>
          <td style="color:{MUTED};font-size:9px;text-transform:uppercase;letter-spacing:1px">Prob Up</td>
          <td style="text-align:right;color:{UP};font-family:monospace;font-size:12px">
            {int(s.get('prob_up_T3',0)*100)}%
          </td>
        </tr>
        <tr>
          <td style="color:{MUTED};font-size:9px;text-transform:uppercase;letter-spacing:1px">Prob Down</td>
          <td style="text-align:right;color:{DOWN};font-family:monospace;font-size:12px">
            {int(s.get('prob_down_T3',0)*100)}%
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Headline -->
  <tr>
    <td colspan="2" style="padding:14px 20px;background:{BG2};border-top:1px solid {BORDER}">
      <div style="color:{MUTED};font-size:9px;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:6px">Headline</div>
      <div style="color:{TEXT};font-size:14px;font-weight:600;line-height:1.5">
        {hl_headline}
      </div>
    </td>
  </tr>

  {desc_block}
  {body_block}

  <!-- Footer -->
  <tr>
    <td colspan="2" style="padding:10px 20px;border-top:1px solid {BORDER}">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="font-size:10px;color:{MUTED}">
            <mark style="background:#14532d;color:#86efac;padding:1px 5px;border-radius:3px;font-size:9px">green</mark>
            = bullish &nbsp;
            <mark style="background:#7f1d1d;color:#fca5a5;padding:1px 5px;border-radius:3px;font-size:9px">red</mark>
            = bearish keyword
          </td>
          <td style="text-align:right">
            <a href="{s.get('url','#')}" style="color:{CYAN};font-size:12px;text-decoration:none;font-weight:600">
              Read article →
            </a>
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>"""


# ── Main send function ────────────────────────────────────────────────────────
def send_signal_email(signals: list[dict]):
    if not EMAIL_ENABLED or not signals:
        return

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    count   = len(signals)
    top     = signals[0]
    subject = (f"[{'↑' if top['direction_T3']=='UP' else '↓'} {top['direction_T3']}] "
               f"{top.get('name', top['asset'])} "
               f"{top['confidence_T3']*100:.0f}% conf"
               + (f" + {count-1} more" if count > 1 else ""))

    cards_html = ""
    for s in signals:
        cards_html += _signal_card(s, scrape_article(s.get("url", "")))

    dashboard_url = "https://news-pulse-dashboard1.onrender.com"

    html = f"""<!DOCTYPE html>
<html><body style="font-family:Arial,Helvetica,sans-serif;background:{BG};color:{TEXT};
                   margin:0;padding:20px 0">
<div style="max-width:680px;margin:0 auto;padding:0 16px">

  <!-- ═══ HEADER ═══ -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:linear-gradient(135deg,{SURFACE},{BG2});
                border:1px solid rgba(0,180,216,.25);border-top:3px solid {CYAN};
                border-radius:14px;margin-bottom:20px;overflow:hidden">
    <tr>
      <td style="padding:22px 26px">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td style="vertical-align:middle">
              <div style="font-size:10px;color:{CYAN};text-transform:uppercase;
                          letter-spacing:2.5px;font-weight:700;margin-bottom:4px">
                ⚡ PULSE Terminal
              </div>
              <div style="font-size:26px;font-weight:900;color:{TEXT};line-height:1.1">
                News Impact Alert
              </div>
              <div style="font-size:12px;color:{MUTED};margin-top:5px">
                {now_str} &bull; {count} high-confidence signal{'s' if count>1 else ''} detected
              </div>
            </td>
            <td style="text-align:right;vertical-align:middle;padding-left:16px">
              <a href="{dashboard_url}"
                 style="background:{CYAN};color:{BG};padding:10px 18px;
                        border-radius:8px;font-size:12px;font-weight:800;
                        text-decoration:none;display:inline-block;white-space:nowrap">
                Open Dashboard →
              </a>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>

  {cards_html}

  <!-- ═══ FOOTER ═══ -->
  <div style="text-align:center;padding:16px;color:{MUTED};font-size:10px">
    Min confidence threshold: {MIN_CONFIDENCE*100:.0f}% &bull;
    <a href="{dashboard_url}" style="color:#334155">{dashboard_url}</a>
  </div>

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
