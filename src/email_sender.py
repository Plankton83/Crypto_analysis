"""
Send the HTML report by email via SMTP.

Required .env variables:
    EMAIL_SMTP_HOST      — e.g. smtp.gmail.com
    EMAIL_SMTP_PORT      — e.g. 587  (STARTTLS) or 465 (SSL)
    EMAIL_SMTP_USER      — sender login / address
    EMAIL_SMTP_PASSWORD  — SMTP password or app-password
    EMAIL_TO             — recipient address (or comma-separated list)
    EMAIL_FROM           — optional, defaults to EMAIL_SMTP_USER
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def send_report(html_path: str, symbol: str) -> None:
    """
    Read the HTML report at *html_path* and send it as an HTML email.

    Raises RuntimeError if required env vars are missing.
    Raises smtplib / socket exceptions on delivery failure.
    """
    host     = os.getenv("EMAIL_SMTP_HOST", "").strip()
    port     = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    user     = os.getenv("EMAIL_SMTP_USER", "").strip()
    password = os.getenv("EMAIL_SMTP_PASSWORD", "").strip()
    to_raw   = os.getenv("EMAIL_TO", "").strip()
    from_addr = os.getenv("EMAIL_FROM", user).strip()

    missing = [k for k, v in [
        ("EMAIL_SMTP_HOST", host),
        ("EMAIL_SMTP_USER", user),
        ("EMAIL_SMTP_PASSWORD", password),
        ("EMAIL_TO", to_raw),
    ] if not v]
    if missing:
        raise RuntimeError(
            f"Cannot send email — missing env vars: {', '.join(missing)}"
        )

    recipients = [r.strip() for r in to_raw.split(",") if r.strip()]

    html_content = Path(html_path).read_text(encoding="utf-8")

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"Crypto Prediction Report — {symbol} — {timestamp}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    if port == 465:
        with smtplib.SMTP_SSL(host, port) as smtp:
            smtp.login(user, password)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    else:
        with smtplib.SMTP(host, port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(user, password)
            smtp.sendmail(from_addr, recipients, msg.as_string())
