import os
from monitoring.alert_system import AlertSystem

# Configuration des variables d'environnement pour le test
os.environ["SENDER_EMAIL"] = ""
os.environ["SENDER_EMAIL_PASSWORD"] = ""
os.environ["RECIPIENT_EMAIL"] = ""

alert_system = AlertSystem()
result = alert_system.send_alert("Test Subject", "Test Message")

print("Email sent:", result)

# Nettoyage des variables d'environnement après le test
del os.environ["SENDER_EMAIL"]
del os.environ["SENDER_EMAIL_PASSWORD"]
del os.environ["RECIPIENT_EMAIL"]
