import smtplib
from email.message import EmailMessage
from config import Config

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not (Config.SMTP_HOST and Config.SMTP_PORT and Config.SMTP_USER and Config.SMTP_PASSWORD and Config.MAIL_FROM):
        print(" Email not configured; skipping send.")
        return False
    try:
        msg = EmailMessage()
        msg["From"] = Config.MAIL_FROM
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f" Failed to send email: {e}")
        return False
