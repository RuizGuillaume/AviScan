import os
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import logging
import tensorflow as tf
import time
import json
import csv
from datetime import datetime
from alert_system import AlertSystem

# On lance le serveur FastAPI
app = FastAPI()

# On instancie la classe qui permet d'envoyer des alertes par email
alert_system = AlertSystem()

# On créer les différents chemins
volume_path = "volume_data"
log_folder = os.path.join(volume_path, "logs")
mlruns_path = os.path.join(volume_path, "mlruns")
prod_model_id_path = os.path.join(mlruns_path, "prod_model_id.txt")
temp_folder = os.path.join(volume_path, "temp_images")

# On créer le dossier si nécessaire
os.makedirs(log_folder, exist_ok=True)

# On configure le logging pour les informations et les erreurs
logging.basicConfig(
    filename=os.path.join(log_folder, "inference.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S %p",
)

# Cette variable s'incrémente dès que le temps d'inférence est trop long
too_long_inference = 0

# ----------------------------------------------------------------------------------------- #


class predictClass:
    """
    Cette classe permet d'effectuer des prédictions à partir d'un modèle .h5 (Keras)
    """

    def __init__(self, model_path, img_size=(224, 224)):
        self.img_size = img_size
        self.model_path = model_path

        # region On créer un dossier pour stocker l'historique des inférences
        volume_path = "volume_data"
        hist_inferences_dir = os.path.join(volume_path, "logs/inferences")
        os.makedirs(hist_inferences_dir, exist_ok=True)

        self.csv_filename = os.path.join(
            hist_inferences_dir, f'inferences_{datetime.now().strftime("%d%m%Y_%H%M")}.csv'
        )
        with open(self.csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            columns = [
                "timestamp",
                "id_model",
                "image_name",
                "classes",
                "scores"
            ]

            writer.writerow(columns)

        # endregion

        # Configurer GPU si disponible
        self.configure_gpu()

        try:
            # On charge le model Keras
            self.model = load_model(os.path.join(model_path, "saved_model.h5"))
            # On charge les labels des classes utilisées durant l'entraînement
            with open(os.path.join(model_path, "classes.json"), "r") as file:
                self.class_names = json.load(file)
            logging.info("Modèle chargé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de l'ouverture du modèle: {str(e)}")
            alert_system.send_alert(
                subject="Erreur lors de l'inférence",
                message=f"Erreur lors de l'ouverture du modèle: {str(e)}",
            )
            raise

    def configure_gpu(self):
        try:
            # Si un GPU est présent, on configure l'utilisation dynamique de la mémoire
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(
                    "GPU(s) configuré(s) pour une utilisation dynamique de la mémoire."
                )
        except RuntimeError as e:
            logging.error(f"Erreur lors de la configuration du GPU : {e}")
            alert_system.send_alert(
                subject="Erreur lors de l'inférence",
                message=f"Erreur lors de la configuration du GPU : {e}",
            )

    def predict(self, image_path):

        try:
            # On charge l'image et on effectue le preprocessing pour EfficientNet
            img = image.load_img(image_path, target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            img_ready = preprocess_input(img_array_expanded_dims)

            # On lance la prédiction sur l'image
            prediction = self.model.predict(img_ready)

            # On récupère la liste des 3 meilleurs scores
            meilleurs_scores = np.flip(np.sort(prediction[0])[-3:])
            # On récupère la liste des index des 3 meilleures classes
            meilleures_classes_index = np.flip(np.argsort(prediction[0])[-3:])
            meilleures_classes = []
            # On récupère les labels des classes
            for index in meilleures_classes_index:
                meilleures_classes.append(self.class_names[str(index)])
            logging.info("Prédiction effectuée avec succès.")
            return meilleures_classes, meilleurs_scores
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction : {str(e)}")
            alert_system.send_alert(
                subject="Erreur lors de l'inférence",
                message=f"Erreur lors de la prédiction : {str(e)}",
            )
            raise


def load_classifier(run_id):
    """
    Permet de charger le modèle en entier pour accélerer les inférences suivantes
    """
    # On attends que le chemin vers le dossier du modèle existe
    model_path = os.path.join(
        volume_path, f"mlruns/157975935045122495/{run_id}/artifacts/model/"
    )
    # On instancie de classifier
    classifier = predictClass(model_path=model_path)
    # On fais la prédiction d'une image pour charger le modèle
    # et accélérer les prochaines inférences
    classifier.predict("./load_image.jpg")
    return classifier


# ----------------------------------------------------------------------------------------- #


# On attends que l'identifiant du modèle à utiliser sois dans le volume
while not os.path.exists(prod_model_id_path):
    time.sleep(1)
# On charge l'identifiant
with open(prod_model_id_path, "r") as file:
    run_id = file.read()

# On charge le classifier pour ne pas le charger à chaque inférence
classifier = load_classifier(run_id)


# ----------------------------------------------------------------------------------------- #


@app.get("/")
def read_root():
    return {"Status": "OK"}


# Cette route permet d'effectuer une prédiction sur une image
@app.get("/predict")
async def predict(file_name: str):
    try:
        global too_long_inference
        # Permet de calculer le temps d'inférence
        start_time = time.time()
        # On récupère la bonne image dans le volume
        image_path = os.path.join(temp_folder, file_name)
        # On lance la prédiction
        meilleures_classes, meilleurs_scores = classifier.predict(image_path)

        # On enregistre la prédiction
        with open(classifier.csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    run_id,
                    file_name,
                    meilleures_classes,
                    meilleurs_scores
                ]
            )

        # On calcule temps qui a été nécessaire
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Temps pour l'inférence : {total_time}")
        if total_time > 1:
            too_long_inference += 1
        if too_long_inference > 3:
            # On envoie un email pour indiquer que les 4 dernières inférences étaient trop longues
            too_long_inference = 0
            logging.error("Lenteur détectée pour l'inférence...")
            alert_system.send_alert(
                subject="Lenteur du container d'inférence",
                message="""Les 4 dernières inférences ont pris plus de
                                    1 seconde à s'éxectuer, il y a un problème de performance.
                                    Merci de vous reporter aux logs.""",
            )

        return {
            "predictions": meilleures_classes,
            "scores": meilleurs_scores.tolist(),
            "filename": file_name,
        }

    except Exception as e:
        logging.error(f"Un problème est survenu lors de l'inférence: {e}")
        alert_system.send_alert(
            subject="Erreur lors de l'inférence",
            message=f"Un problème est survenu lors de l'inférence: {e}",
        )
        raise HTTPException(
            status_code=500, detail=f"Un problème est survenu lors de l'inférence: {e}"
        )


@app.post("/switchmodel")
async def switch_model(run_id: str = Body(...)):
    try:
        global classifier
        # On récupère le run_id et on l'enregistre
        run_id = run_id.removeprefix("run_id=")
        # On recharge le classifier avec le nouveau modèle
        classifier = load_classifier(run_id)
        with open(os.path.join(mlruns_path, "prod_model_id.txt"), "w") as file:
            file.write(run_id)
        logging.info("Changement de modèle effectué !")
        return {
            f"Le nouveau modèle utilisé provient maintenant du run id suivant : {run_id}"
        }

    except Exception as e:
        logging.error(f"Le changement de modèle n'a pas fonctionné : {e}")
        alert_system.send_alert(
            subject="Erreur lors de l'inférence",
            message=f"Le changement de modèle n'a pas fonctionné : {e}",
        )
        raise HTTPException(
            status_code=500, detail=f"Le changement de modèle n'a pas fonctionné : {e}"
        )
