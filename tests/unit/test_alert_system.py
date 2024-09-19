import unittest
import os
from monitoring.alert_system import AlertSystem
import logging

# Configuration du logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_logger")


class TestAlertSystem(unittest.TestCase):

    def setUp(self):
        # Assurez-vous que ces variables d'environnement sont définies avant d'exécuter les tests
        self.alert_email = os.getenv("SENDER_EMAIL")
        self.email_password = os.getenv("SENDER_EMAIL_PASSWORD")
        self.recipient_email = os.getenv("RECIPIENT_EMAIL")

        if not all([self.alert_email, self.email_password, self.recipient_email]):
            raise ValueError("Les variables d'environnement nécessaires ne sont pas définies.")

        self.alert_system = AlertSystem()

    def test_send_alert(self):
        subject = "Test Alert"
        message = "This is a test alert message."
        result = self.alert_system.send_alert(subject, message)

        logger.info(f"Résultat de l'envoi de l'alerte : {result}")

        self.assertTrue(result, "L'envoi de l'alerte a échoué")

        # Vous devrez vérifier manuellement que l'email a été reçu

    def test_send_alert_with_invalid_credentials(self):
        # Sauvegardez le mot de passe correct
        correct_password = os.environ["SENDER_EMAIL_PASSWORD"]

        # Remplacez temporairement par un mot de passe invalide
        os.environ["SENDER_EMAIL_PASSWORD"] = "invalid_password"

        # Réinitialisez l'AlertSystem pour qu'il utilise le nouveau mot de passe
        self.alert_system = AlertSystem()

        subject = "Test Alert with Invalid Credentials"
        message = "This alert should not be sent."
        result = self.alert_system.send_alert(subject, message)

        logger.info(f"Résultat de l'envoi de l'alerte avec des identifiants invalides : {result}")

        self.assertFalse(result, "L'alerte a été envoyée malgré des identifiants invalides")

        # Restaurez le mot de passe correct
        os.environ["SENDER_EMAIL_PASSWORD"] = correct_password


if __name__ == "__main__":
    unittest.main()
