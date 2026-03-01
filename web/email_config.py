"""
Email alert configuration.
Fill in your credentials below, then restart the server.

For Gmail: go to https://myaccount.google.com/apppasswords
  (2-Step Verification must be ON)
  Create an App Password → copy the 16-char code → paste as SENDER_PASSWORD
"""

# ── Who gets the alerts ───────────────────────────────────────────────────────
RECIPIENT_EMAIL = "YOUR_EMAIL@gmail.com"   # <-- change this

# ── Sending account ───────────────────────────────────────────────────────────
SENDER_EMAIL    = "YOUR_EMAIL@gmail.com"   # <-- change this (your Gmail)
SENDER_PASSWORD = "xxxx xxxx xxxx xxxx"    # <-- 16-char Gmail App Password

# ── SMTP server (change only if not Gmail) ────────────────────────────────────
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# ── Alert thresholds ──────────────────────────────────────────────────────────
# Only send email when confidence >= this value AND direction is UP or DOWN
MIN_CONFIDENCE = 0.62   # 62%
