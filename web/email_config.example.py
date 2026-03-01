"""
Email alert configuration — copy this file to email_config.py and fill in your details.

For Gmail App Password: https://myaccount.google.com/apppasswords
  (2-Step Verification must be ON)
"""

RECIPIENT_EMAIL = "your_email@gmail.com"   # where to receive alerts
SENDER_EMAIL    = "your_email@gmail.com"   # your Gmail (or Google Workspace)
SENDER_PASSWORD = "xxxx xxxx xxxx xxxx"    # 16-char Gmail App Password

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# Only alert when confidence >= this AND direction is UP or DOWN
MIN_CONFIDENCE = 0.62
