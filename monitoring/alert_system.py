import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
from app.utils.logger import setup_logger

load_dotenv()
logger = setup_logger("alert_system", "logs/alert_system.log")


class AlertSystem:
    def __init__(self):
        self.from_email = os.getenv("SENDER_EMAIL")
        self.password = os.getenv("SENDER_EMAIL_PASSWORD")
        self.to_email = os.getenv("RECIPIENT_EMAIL")
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 465  # Port pour SSL

    def send_alert(self, subject, message):
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = self.to_email

        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.from_email, self.password)
                server.sendmail(self.from_email, self.to_email, msg.as_string())
            logger.info("Alerte envoyée avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'alerte : {str(e)}")
            return False
