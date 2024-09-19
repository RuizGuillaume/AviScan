import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

DRIFT_THRESHOLDS = {
    "class_increase": 0.05,
    "new_class": max(20, 0.01),
    "confidence_decrease": 0.05,
}

EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": os.getenv("SENDER_EMAIL"),
    "receiver_email": os.getenv("RECIPIENT_EMAIL"),
    "password": os.getenv("SENDER_EMAIL_PASSWORD"),
}
