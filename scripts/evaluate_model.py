import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import mlflow
from app.utils.logger import setup_logger

logger = setup_logger("evaluate_model", "evaluate_model.log")


def load_test_data(test_path, img_size=(224, 224), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return test_generator


def get_latest_model(base_path):
    model_folders = glob.glob(os.path.join(base_path, "saved_model*"))
    if not model_folders:
        return None
    latest_folder = max(model_folders, key=os.path.getmtime)
    return latest_folder


def evaluate_model():
    logger.info("Début de l'évaluation du modèle")

    # Définition des chemins
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, "models")
    test_path = os.path.join(BASE_DIR, "data", "test")

    logger.info(f"Dossier des modèles : {models_dir}")
    logger.info(f"Dossier de test : {test_path}")

    # Trouver le modèle le plus récent
    model_path = get_latest_model(models_dir)
    if not model_path:
        logger.error("Aucun modèle sauvegardé trouvé.")
        return

    logger.info(f"Modèle le plus récent trouvé : {model_path}")

    try:
        logger.info("Tentative de chargement du modèle...")
        # Charger le SavedModel
        loaded_model = tf.saved_model.load(model_path)
        infer = loaded_model.signatures["serving_default"]
        logger.info("SavedModel chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return

    logger.info("Chargement des données de test...")
    test_generator = load_test_data(test_path)
    logger.info(f"Données de test chargées. Nombre de classes : {len(test_generator.class_indices)}")

    logger.info("Génération des prédictions...")
    all_predictions = []
    all_true_classes = []

    for i in range(len(test_generator)):
        batch_images, batch_labels = test_generator[i]
        batch_predictions = infer(tf.constant(batch_images))[
            "dense_2"
        ]  # Assurez-vous que 'dense_2' est le bon nom de la couche de sortie
        all_predictions.append(batch_predictions.numpy())
        all_true_classes.extend(np.argmax(batch_labels, axis=1))

    predictions = np.concatenate(all_predictions)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.array(all_true_classes)

    logger.info("Calcul des métriques...")
    accuracy = np.mean(predicted_classes == true_classes)
    logger.info(f"Précision sur l'ensemble de test : {accuracy:.4f}")

    logger.info("Calcul du rapport de classification...")
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_names)
    logger.info("Rapport de classification :\n" + report)

    logger.info("Calcul de la matrice de confusion...")
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    logger.info("Matrice de confusion :\n" + str(conf_matrix))

    logger.info("Enregistrement des résultats dans MLflow...")
    mlflow.set_experiment("Bird Classification Evaluation")
    with mlflow.start_run():
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    logger.info("Calcul des métriques par classe...")
    for i, class_name in enumerate(class_names):
        class_precision = precision_score(true_classes == i, predicted_classes == i, average="binary")
        class_recall = recall_score(true_classes == i, predicted_classes == i, average="binary")
        class_f1 = f1_score(true_classes == i, predicted_classes == i, average="binary")

        logger.info(f"  {class_name}:")
        logger.info(f"    Précision: {class_precision:.4f}")
        logger.info(f"    Rappel: {class_recall:.4f}")
        logger.info(f"    F1-Score: {class_f1:.4f}")

        mlflow.log_metric(f"precision_{class_name}", class_precision)
        mlflow.log_metric(f"recall_{class_name}", class_recall)
        mlflow.log_metric(f"f1_score_{class_name}", class_f1)

    logger.info("Évaluation terminée.")
    logger.info("Vérifiez les résultats dans MLflow et dans ce fichier de log.")
