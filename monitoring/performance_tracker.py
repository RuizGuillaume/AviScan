import pandas as pd
from datetime import datetime
from app.utils.data_manager import DataManager
from app.utils.logger import setup_logger

logger = setup_logger("performance_tracker", "performance_tracker.log")


class PerformanceTracker:
    def __init__(self):
        self.data_manager = DataManager()
        self.class_names = self.data_manager.get_class_names()
        self.current_log_file = None
        logger.info(f"PerformanceTracker initialized with {len(self.class_names)} classes")

    def get_current_log_file(self):
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"logs/performance_logs_{timestamp}.csv"

    def log_prediction(self, predicted_class, confidence, true_class=None):
        log_file = self.get_current_log_file()

        new_log = pd.DataFrame(
            {
                "date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "predicted_class": [predicted_class],
                "confidence": [confidence],
                "true_class": [true_class],
            }
        )

        try:
            if log_file != self.current_log_file:
                self.current_log_file = log_file
                new_log.to_csv(log_file, index=False)
            else:
                new_log.to_csv(log_file, mode="a", header=False, index=False)
            logger.info(
                f"Prédiction enregistrée : {predicted_class}, confidence: {confidence}, true_class: {true_class}"
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la prédiction: {str(e)}")

    def get_performance_metrics(self, log_file=None):
        if log_file is None:
            log_file = self.get_current_log_file()

        logger.info(f"Tentative de lecture du fichier de log : {log_file}")

        try:
            df = pd.read_csv(log_file)
            logger.info(f"Fichier de log lu avec succès. Nombre de lignes : {len(df)}")
        except FileNotFoundError:
            logger.warning(f"Fichier de logs {log_file} non trouvé.")
            return None, {}

        if df.empty:
            logger.warning("Le fichier de logs est vide.")
            return None, {}

        # Filtrer les entrées sans vraie classe
        df = df.dropna(subset=["true_class"])

        if df.empty:
            logger.warning("Aucune entrée avec une vraie classe n'a été trouvée.")
            return None, {}

        overall_accuracy = (df["predicted_class"] == df["true_class"]).mean()
        logger.info(f"Précision globale calculée : {overall_accuracy}")

        class_accuracies = {}
        for class_name in self.class_names:
            class_df = df[df["true_class"] == class_name]
            if not class_df.empty:
                class_accuracies[class_name] = (class_df["predicted_class"] == class_df["true_class"]).mean()
            else:
                class_accuracies[class_name] = None

        logger.info(f"Précisions par classe calculées : {class_accuracies}")
        return overall_accuracy, class_accuracies
