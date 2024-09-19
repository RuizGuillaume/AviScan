import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_manager import DataManager
from app.models.predictClass import predictClass
from monitoring.performance_tracker import PerformanceTracker
from app.utils.logger import setup_logger
from collections import Counter

logger = setup_logger("test_prediction_logging", "test_prediction_logging.log")


def test_prediction_and_logging():
    data_manager = DataManager()
    predictor = predictClass()
    performance_tracker = PerformanceTracker()

    new_data = data_manager.load_new_data()

    # Mélanger les données et prendre un échantillon plus grand
    random.shuffle(new_data)
    sample_size = min(100, len(new_data))

    for image_path, true_class in new_data[:sample_size]:
        class_name, confidence = predictor.predict(image_path)
        performance_tracker.log_prediction(class_name, confidence, true_class=true_class)
        logger.info(
            f"Image: {image_path}, Vraie classe: {true_class}, Prédiction: {class_name}, Confiance: {confidence}"
        )

    # Vérifier les métriques
    overall_accuracy, class_accuracies = performance_tracker.get_performance_metrics()
    logger.info(f"Précision globale : {overall_accuracy}")
    logger.info("Précisions par classe :")
    for class_name, accuracy in class_accuracies.items():
        if accuracy is not None:
            logger.info(f"  {class_name}: {accuracy:.4f}")

    # Afficher des statistiques sur les classes prédites
    predicted_classes = [pred for pred, true in performance_tracker.predictions]
    true_classes = [true for pred, true in performance_tracker.predictions]

    pred_counts = Counter(predicted_classes)
    true_counts = Counter(true_classes)

    logger.info("Nombre de prédictions par classe :")
    for class_name, count in pred_counts.items():
        logger.info(f"  {class_name}: {count}")

    logger.info("Nombre d'instances réelles par classe :")
    for class_name, count in true_counts.items():
        logger.info(f"  {class_name}: {count}")


if __name__ == "__main__":
    test_prediction_and_logging()
