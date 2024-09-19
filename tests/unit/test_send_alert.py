import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

load_dotenv()

from_email = os.getenv("SENDER_EMAIL")
password = os.getenv("SENDER_EMAIL_PASSWORD")
to_email = os.getenv("RECIPIENT_EMAIL")
smtp_server = "smtp.gmail.com"
smtp_port = 465  # Port pour SSL


def send_alert(subject, message):
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.set_debuglevel(1)  # Afficher les logs de débogage SMTP
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print("Alerte envoyée avec succès")
        return True
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'alerte : {str(e)}")
        return False


# Test en dehors du cadre unitaire
if __name__ == "__main__":
    send_alert("Test Subject", "Test Message")
