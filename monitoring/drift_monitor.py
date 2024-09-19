import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from app.utils.logger import setup_logger

logger = setup_logger("drift_monitor", "drift_monitor.log")


class DriftMonitor:
    def __init__(self, train_data_path="data/train"):
        self.train_data_path = train_data_path
        self.initial_class_counts = self.get_initial_class_counts()
        self.average_class_size = np.mean(list(self.initial_class_counts.values())) if self.initial_class_counts else 0
        self.class_increase_threshold = 1.05  # 5% d'augmentation
        self.new_class_threshold = max(10, int(0.03 * self.average_class_size))
        self.confidence_drop_threshold = 0.03

    def get_initial_class_counts(self):
        class_counts = {}
        for class_name in os.listdir(self.train_data_path):
            class_path = os.path.join(self.train_data_path, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(
                    [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                )
        logger.info(f"Comptages initiaux des classes : {class_counts}")
        return class_counts

    def get_current_log_file(self):
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"logs/performance_logs_{timestamp}.csv"

    def check_drift(self, log_file=None):
        if log_file is None:
            log_file = self.get_current_log_file()

        if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
            logger.warning(f"Le fichier de log {log_file} n'existe pas ou est vide.")
            return False, "Pas assez de données pour détecter un drift"

        df = pd.read_csv(log_file)
        drift_detected = False
        drift_reasons = []

        # Calculer les comptes de classes à partir du fichier de log
        current_class_counts = df["predicted_class"].value_counts().to_dict()
        logger.info(f"Comptages actuels des classes : {current_class_counts}")

        for class_name, initial_count in self.initial_class_counts.items():
            current_count = current_class_counts.get(class_name, 0)
            if current_count > initial_count * self.class_increase_threshold:
                drift_detected = True
                drift_reasons.append(
                    f"La classe {class_name} a augmenté de plus de 5%: {initial_count} à {current_count}"
                )

        new_classes = set(current_class_counts.keys()) - set(self.initial_class_counts.keys())
        for new_class in new_classes:
            new_class_count = current_class_counts[new_class]
            if new_class_count >= self.new_class_threshold:
                drift_detected = True
                drift_reasons.append(f"Nouvelle classe détectée: {new_class} avec {new_class_count} images")

        recent_df = df[df["date"] >= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")]
        if not recent_df.empty:
            recent_confidence = recent_df["confidence"].mean()
            past_confidence = df[df["date"] < recent_df["date"].min()]["confidence"].mean()
            logger.info(f"Confiance récente: {recent_confidence}, Confiance passée: {past_confidence}")

            if recent_confidence < past_confidence - self.confidence_drop_threshold:
                drift_detected = True
                drift_reasons.append(f"Baisse de confiance: {past_confidence:.2f} à {recent_confidence:.2f}")

        if drift_detected:
            logger.warning(f"Drift détecté: {drift_reasons}")
        else:
            logger.info("Aucun drift détecté")

        return drift_detected, (drift_reasons if drift_detected else "Aucun drift détecté")
