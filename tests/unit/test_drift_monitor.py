import unittest
import pandas as pd
from monitoring.drift_monitor import DriftMonitor
import os
from datetime import datetime, timedelta


class TestDriftMonitor(unittest.TestCase):
    def setUp(self):
        self.test_train_data_path = "test_train_data"
        self.test_log_file = "test_performance_logs.csv"

        # Créer un répertoire de données d'entraînement factice
        os.makedirs(os.path.join(self.test_train_data_path, "class1"), exist_ok=True)
        os.makedirs(os.path.join(self.test_train_data_path, "class2"), exist_ok=True)
        for i in range(100):
            open(os.path.join(self.test_train_data_path, "class1", f"img{i}.jpg"), "w").close()
            open(os.path.join(self.test_train_data_path, "class2", f"img{i}.jpg"), "w").close()

        # Créer un fichier CSV de test avec une augmentation de class1 et class2
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        test_data = pd.DataFrame(
            {
                "date": [today] * 230 + [yesterday] * 220,
                "predicted_class": ["class1"] * 230 + ["class2"] * 220,
                "confidence": [0.9] * 230 + [0.8] * 220,
            }
        )
        test_data.to_csv(self.test_log_file, index=False)

    def tearDown(self):
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
        for root, dirs, files in os.walk(self.test_train_data_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_train_data_path)

    def test_check_drift(self):
        monitor = DriftMonitor(train_data_path=self.test_train_data_path)
        drift_detected, reasons = monitor.check_drift(log_file=self.test_log_file)

        print(f"Drift detected: {drift_detected}")
        print(f"Reasons: {reasons}")
        print(f"Initial class counts: {monitor.initial_class_counts}")

        # Calculer les comptages actuels à partir du fichier de log
        df = pd.read_csv(self.test_log_file)
        current_counts = df["predicted_class"].value_counts().to_dict()
        print(f"Current class counts: {current_counts}")

        self.assertTrue(drift_detected)
        self.assertIn("La classe class1 a augmenté de plus de 5%: 100 à 230", reasons)
        self.assertIn("La classe class2 a augmenté de plus de 5%: 100 à 220", reasons)
        self.assertEqual(monitor.initial_class_counts, {"class1": 100, "class2": 100})
        self.assertEqual(current_counts, {"class1": 230, "class2": 220})

    def test_check_drift_no_data(self):
        empty_log_file = "empty_log.csv"
        open(empty_log_file, "w").close()  # Créer un fichier vide

        monitor = DriftMonitor(train_data_path=self.test_train_data_path)
        drift_detected, reason = monitor.check_drift(log_file=empty_log_file)

        self.assertFalse(drift_detected)
        self.assertEqual(reason, "Pas assez de données pour détecter un drift")

        os.remove(empty_log_file)


if __name__ == "__main__":
    unittest.main()
