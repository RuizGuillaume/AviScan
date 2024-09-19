import sys
import os
import threading
import time
import queue
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessing"))

import mlflow
from monitoring.drift_monitor import DriftMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alert_system import AlertSystem
from monitoring.system_monitor import SystemMonitor
from app.utils.logger import setup_logger, clean_old_logs
from app.utils.data_manager import DataManager
from app.models.predictClass import predictClass
from training.train_model import train_model
from preprocessing.preprocess_dataset import CleanDB
from app.utils.data_version_manager import DataVersionManager
from scripts.downloadDataset import download_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = setup_logger("pipeline", "pipeline.log")


def preprocess_data(data_path, test_dataset_mode: bool = False):
    logger.info("Début du prétraitement des données")
    cleaner = CleanDB(data_path, treshold=False, test_mode=test_dataset_mode)
    cleaner.cleanAll()
    logger.info("Prétraitement des données terminé")

    data_version_manager = DataVersionManager(data_path)
    new_version = data_version_manager.update_version()
    logger.info(f"Nouvelle version des données : {new_version}")
    return new_version


class SystemMonitorThread(threading.Thread):
    def __init__(self, duration):
        threading.Thread.__init__(self)
        self.duration = duration
        self.stop_event = threading.Event()
        self.monitor = SystemMonitor()
        self.metrics_queue = queue.Queue()

    def run(self):
        start_time = time.time()
        while not self.stop_event.is_set() and time.time() - start_time < self.duration:
            metrics = self.monitor.get_metrics()
            self.metrics_queue.put((time.time(), metrics))
            time.sleep(5)

    def stop(self):
        self.stop_event.set()

    def log_metrics(self):
        while not self.metrics_queue.empty():
            timestamp, metrics = self.metrics_queue.get()
            self.monitor.log_metrics(metrics, datetime.fromtimestamp(timestamp))


def run_pipeline(test_dataset_mode: bool = False):
    clean_old_logs()

    mlflow.end_run()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Bird Classification Project_{timestamp}"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    logger.info(f"Création de l'expérience : {experiment_name}")
    logger.info(f"ID de l'expérience : {experiment_id}")

    with mlflow.start_run(experiment_id=experiment_id, run_name="Main Pipeline Run"):
        mlflow.set_tag("run_type", "pipeline")
        logger.info("Démarrage de la pipeline")

        monitor_thread = SystemMonitorThread(3600)
        monitor_thread.start()

        try:

            data_path = os.path.join(BASE_DIR, "data")
            # Téléchargement du dataset Kaggle
            download_dataset(dataset_name="gpiosenka/100-bird-species", destination_folder=data_path)
            # Préprocessing
            data_version = preprocess_data(data_path=data_path, test_dataset_mode=test_dataset_mode)

            data_manager = DataManager()
            drift_monitor = DriftMonitor()
            performance_tracker = PerformanceTracker()
            alert_system = AlertSystem()

            mlflow.set_tag("data_version", data_version)

            logger.info("Début de l'entraînement du modèle")
            model, drift_detected_during_training = train_model(
                start_mlflow_run=False,
                data_version=data_version,
                experiment_id=experiment_id,
            )
            logger.info("Fin de l'entraînement du modèle")

            monitor_thread.log_metrics()

            if drift_detected_during_training:
                logger.warning("Drift détecté pendant l'entraînement")
                mlflow.log_param("drift_detected_during_training", True)

            predictor = predictClass()

            new_data = data_manager.load_new_data()
            all_classes = data_manager.get_class_names()

            predictions = {class_name: 0 for class_name in all_classes}
            new_species = set()
            unknown_images = []

            logger.info(f"Traitement de {len(new_data)} nouvelles images")

            for image_path, true_class in new_data:
                class_name, confidence = predictor.predict(image_path)

                if class_name in predictions:
                    predictions[class_name] += 1
                else:
                    new_species.add(class_name)
                    predictions[class_name] = 1

                if confidence < 0.5:
                    unknown_images.append(image_path)

                performance_tracker.log_prediction(class_name, confidence, true_class=true_class)

            logger.info(f"Nombre total d'images traitées : {sum(predictions.values())}")
            mlflow.log_metric("total_images_processed", sum(predictions.values()))

            logger.info(f"Nouvelles espèces détectées : {len(new_species)}")
            mlflow.log_metric("new_species_detected", len(new_species))

            logger.info(f"Images non identifiées : {len(unknown_images)}")
            mlflow.log_metric("unknown_images", len(unknown_images))

            for class_name, count in predictions.items():
                if count > 0:
                    logger.info(f"  {class_name}: {count} images")
                    mlflow.log_metric(f"predictions_{class_name}", count)

            drift_detected, drift_details = drift_monitor.check_drift()
            drift_detected = drift_detected or drift_detected_during_training

            if drift_detected:
                logger.warning(f"Drift détecté: {drift_details}")
                alert_message = (
                    f"Drift détecté dans la pipeline de reconnaissance d'oiseaux.\n\nDétails : {drift_details}"
                )
                alert_system.send_alert("Alerte de Drift", alert_message)
                mlflow.log_param("drift_detected", True)
                mlflow.log_param("drift_details", drift_details)
            else:
                logger.info("Aucun drift détecté")
                mlflow.log_param("drift_detected", False)

            overall_accuracy, class_accuracies = performance_tracker.get_performance_metrics()
            if overall_accuracy is not None:
                logger.info(f"Précision globale : {overall_accuracy:.4f}")
                mlflow.log_metric("overall_accuracy", overall_accuracy)
            else:
                logger.warning("Impossible de calculer la précision globale")

            logger.info("Précision par classe :")
            for class_name in all_classes:
                accuracy = class_accuracies.get(class_name)
                if accuracy is not None:
                    logger.info(f"  {class_name}: {accuracy:.4f}")
                    mlflow.log_metric(f"accuracy_{class_name}", accuracy)
                else:
                    logger.info(f"  {class_name}: Pas de données")

            logger.info("Pipeline terminée")

        finally:
            monitor_thread.stop()
            monitor_thread.join()
            monitor_thread.log_metrics()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline Script")  # Créer un parseur d'arguments
    parser.add_argument(
        "--test_dataset_mode",
        type=bool,
        default=False,
        help="Passez test_dataset_mode à True pour générer un dataset allégé",
    )  # Ajouter un argument pour --test_dataset_mode
    args = parser.parse_args()
    run_pipeline(test_dataset_mode=args.test_dataset_mode)
