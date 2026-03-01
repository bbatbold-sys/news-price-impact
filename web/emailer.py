"""
Sends email alerts for high-confidence signals.
"""
import smtplib, logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone

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


def _arrow(direction: str) -> str:
    return {"UP": "UP", "DOWN": "DOWN", "FLAT": "FLAT"}.get(direction, direction)


def _signal_color(direction: str) -> str:
    return {"UP": "#22c55e", "DOWN": "#ef4444", "FLAT": "#94a3b8"}.get(direction, "#94a3b8")


def send_signal_email(signals: list[dict]):
    """
    signals: list of dicts with keys:
      asset, name, asset_class, headline, direction_T3, confidence_T3,
      pred_T3, sentiment, time_to_react
    Only called for high-confidence non-FLAT signals.
    """
    if not EMAIL_ENABLED or not signals:
        return

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    count   = len(signals)
    subject = f"[News Alert] {count} new signal{'s' if count > 1 else ''} — {now_str}"

    # ── Build HTML rows ───────────────────────────────────────────────────────
    rows_html = ""
    for s in signals:
        color     = _signal_color(s["direction_T3"])
        ret_str   = f"{s['pred_T3']:+.2f}%" if s.get("pred_T3") is not None else "N/A"
        conf_pct  = f"{s['confidence_T3']*100:.0f}%" if s.get("confidence_T3") else "N/A"
        rows_html += f"""
        <tr>
          <td style="padding:10px;font-weight:bold">{s.get('name', s['asset'])} ({s['asset']})</td>
          <td style="padding:10px;color:{color};font-weight:bold;font-size:18px">
              {_arrow(s['direction_T3'])}
          </td>
          <td style="padding:10px">{conf_pct}</td>
          <td style="padding:10px">{ret_str}</td>
          <td style="padding:10px;color:#64748b">{s.get('sentiment','')}</td>
          <td style="padding:10px;color:#f59e0b">{s.get('time_to_react','')}</td>
          <td style="padding:10px;font-size:12px;color:#475569">{s.get('headline','')[:120]}</td>
        </tr>"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:20px">
      <div style="max-width:900px;margin:auto">
        <h2 style="color:#38bdf8;margin-bottom:4px">News Impact Alert</h2>
        <p style="color:#64748b;margin-top:0">{now_str} &bull; {count} high-confidence signal{'s' if count > 1 else ''}</p>
        <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden">
          <thead>
            <tr style="background:#0f172a;color:#94a3b8;font-size:12px;text-transform:uppercase">
              <th style="padding:10px;text-align:left">Asset</th>
              <th style="padding:10px;text-align:left">Signal</th>
              <th style="padding:10px;text-align:left">Confidence</th>
              <th style="padding:10px;text-align:left">Est. Return</th>
              <th style="padding:10px;text-align:left">Sentiment</th>
              <th style="padding:10px;text-align:left">React In</th>
              <th style="padding:10px;text-align:left">Headline</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        <p style="color:#475569;font-size:12px;margin-top:16px">
          Dashboard: <a href="http://localhost:5000" style="color:#38bdf8">http://localhost:5000</a>
          &bull; Min confidence threshold: {MIN_CONFIDENCE*100:.0f}%
        </p>
      </div>
    </body></html>
    """

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
