import os
import pandas as pd
import mlflow
import mlflow.keras
import numpy as np
import logging
import mlflow
import shutil
import json
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from fastapi import FastAPI, HTTPException, BackgroundTasks
from mlflow.tracking import MlflowClient
from alert_system import AlertSystem

# On lance le serveur FastAPI
app = FastAPI()

# On instancie la classe qui permet d'envoyer des alertes par email
alert_system = AlertSystem()

# On créer les différents chemins
volume_path = "volume_data"
log_folder = os.path.join(volume_path, "logs")
state_folder = os.path.join(volume_path, "containers_state")
dataset_folder = os.path.join(volume_path, "dataset_clean")
state_path = os.path.join(state_folder, "training_state.txt")
preprocessing_state_path = os.path.join(state_folder, "preprocessing_state.txt")
drift_monitor_state_path = os.path.join(state_folder, "drift_monitor_state.txt")
mlruns_path = os.path.join(volume_path, "mlruns")
train_path = os.path.join(dataset_folder, "train")
valid_path = os.path.join(dataset_folder, "valid")
test_path = os.path.join(dataset_folder, "test")

# On créer les dossiers si nécessaire
os.makedirs(state_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# On déclare le nom de l'expérience MLflow à récupérer
experiment_id = "157975935045122495"

# On déclare l'état par défaut du container
with open(state_path, "w") as file:
    file.write("0")

# On configure le logging pour les informations et les erreurs
logging.basicConfig(
    filename=os.path.join(log_folder, "training.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S %p",
)

# On ajoute le dossier mlruns contenant un run complet s'il n'existe pas dans le volume
if not os.path.exists(mlruns_path):
    shutil.copytree("./mlruns", mlruns_path)
    shutil.copy("./prod_model_id.txt", mlruns_path)
else:
    shutil.rmtree("./mlruns")

# On indique à MLFlow d'effectuer son tracking dans le dossier en question
mlflow.set_tracking_uri("file:///home/app/volume_data/mlruns")
client = MlflowClient()

# ----------------------------------------------------------------------------------------- #


def generate_confusion_matrix(test_generator, model):
    """
    Génére la matrice de confusion (et métriques de recall, precision, et f1-score) pour le modèle
    """
    try:
        # On lance une prédiction sur tout le générateur de test
        predictions = model.predict(test_generator)
        # On récupère les index des classes prédites
        predicted_classes = np.argmax(predictions, axis=1)

        # On récupère les labels correspondants aux indexs
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        # On créer la matrice de confusion
        conf_matrix = confusion_matrix(true_classes, predicted_classes)

        # On ajoute les metriques dans un DataFrame
        confusion_df = pd.DataFrame(
            conf_matrix, index=class_labels, columns=class_labels
        )
        confusion_df = add_metrics(confusion_df)

        # On enregistre la matrice de confusion
        confusion_df.to_csv("./initial_confusion_matrix.csv")
        mlflow.log_artifact("./initial_confusion_matrix.csv")
        os.remove("./initial_confusion_matrix.csv")
    except Exception as e:
        logging.error(
            f"Un problème est survenu lors de la création de la matrice de confusion : {e}"
        )
        alert_system.send_alert(
            subject="Erreur lors de l'entraînement",
            message=f"Un problème est survenu lors de la création de la matrice de confusion : {e}",
        )


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajout au DataFrame des métriques de precision, recall et f1-score
    """
    try:
        # On calcule la precision (VP/VP+FP) pour chaque classe et ajoute la colonne
        df["Precision"] = df.apply(
            lambda row: df.loc[row.name, row.name] / df[row.name].sum()
            if df[row.name].sum() != 0
            else 0,
            axis=1
        )
        # On calcule le recall (VP/VP+FN) pour chaque classe et ajoute la colonne
        df["Recall"] = df.apply(
            lambda row: df.loc[row.name, row.name] / df.loc[row.name].sum()
            if df.loc[row.name].sum() != 0
            else 0,
            axis=1
        )
        # On calcule le f1-score ((2*Rec*Pre)/(Rec+Pre)) pour chaque classe et ajoute la colonne
        df["f1-score"] = df.apply(
            lambda row: (2 * row["Precision"] * row["Recall"]) / (row["Precision"] + row["Recall"])
            if (row["Precision"] + row["Recall"]) != 0
            else 0,
            axis=1
        )

        return df
    except Exception as e:
        logging.error(
            f"Un problème est survenu lors de la création des scores f1, recall et precision : {e}"
        )
        alert_system.send_alert(
            subject="Erreur lors de l'entraînement",
            message=f"Un problème est survenu lors de la création des scores f1, recall et precision : {e}",
        )


def get_worst_f1_scores(run_id: str):
    """
    Renvoie les f1-score et index les plus bas de la matrice de confusion d'une run
    """
    try:
        # Lit la matrice de confusion
        df = pd.read_csv(
            f"{mlruns_path}/{experiment_id}/{run_id}/artifacts/initial_confusion_matrix.csv",
            index_col=0,
        )
        # Récupère les 10 pires scores F1
        worst_values = df.nsmallest(10, "f1-score")

        # On fait correspondre les scores avec les classes
        index_and_values = worst_values.index, worst_values["f1-score"]
        return index_and_values
    except Exception as e:
        logging.error(
            f"Un problème est survenu lors de la récupération des pires classes : {e}"
        )
        alert_system.send_alert(
            subject="Erreur lors de l'entraînement",
            message=f"Un problème est survenu lors de la récupération des pires classes : {e}",
        )


def train_model():
    """
    Fonction qui lance l'entraînement du modèle tout en faisant un suivi avec MLFlow
    """
    try:
        # On indique que l'état du container passe à actif
        with open(state_path, "w") as file:
            file.write("1")

        # On indique le nom de l'expérience dans laquelle se situer
        mlflow.set_experiment("Bird Classification Training")

        # On lance le tracking de la run via MLFlow
        with mlflow.start_run():

            logging.info("Démmarage de l'entraînement")

            # On demande a MLFLow de logger automatiquement les métriques pertinentes
            # mais sans le modèle (qu'on log plus tard manuellement)
            mlflow.keras.autolog(log_models=False)

            # Définition et log de la batch size
            batch_size = 16
            mlflow.log_param("batch_size", batch_size)

            # Définition des callbacks
            reduce_learning_rate = ReduceLROnPlateau(
                monitor="val_loss",
                patience=2,
                min_delta=0.01,
                factor=0.1,
                cooldown=4,
                verbose=1,
            )
            early_stopping = EarlyStopping(
                patience=5, min_delta=0.01, verbose=1, mode="min", monitor="val_loss"
            )

            # Création des générateurs d'images avec augmentation des données
            train_datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode="nearest",
            )
            train_generator = train_datagen.flow_from_directory(
                train_path, target_size=(224, 224), batch_size=batch_size
            )
            valid_generator = ImageDataGenerator().flow_from_directory(
                valid_path, target_size=(224, 224), batch_size=batch_size
            )
            test_generator = ImageDataGenerator().flow_from_directory(
                test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False
            )

            # Récupération du nombre de classes et leurs indexs
            num_classes = train_generator.num_classes
            indices_classes_raw = train_generator.class_indices
            # On inverse les indexs et clés pour créer un dictionnaire pour plus tard
            indices_classes = {}
            for classe in indices_classes_raw:
                indices_classes[indices_classes_raw[classe]] = classe
            classes_file_path = "./classes.json"
            # On enregistre le dictionnaire et on le log dans les artefacts MLflow
            with open(classes_file_path, "w") as json_file:
                json.dump(indices_classes, json_file)
            mlflow.log_artifact(classes_file_path, artifact_path="model")
            os.remove(classes_file_path)

            # On log manuellement le nombre de classes
            mlflow.log_param("num_classes", num_classes)

            # On se base sur le modèle pré-entrainé EfficientNetB0
            base_model = EfficientNetB0(weights="imagenet", include_top=False)

            # On dégèle uniquement les 20 dernières couches pour affiner le modèle
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            for layer in base_model.layers[-20:]:
                layer.trainable = True

            # On ajoute nos couches
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1280, activation="relu")(x)
            x = Dropout(rate=0.2)(x)
            x = Dense(640, activation="relu")(x)
            x = Dropout(rate=0.2)(x)
            predictions = Dense(num_classes, activation="softmax")(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            # On compile le modèle avec un optimiseur Adam et un learning rate adaptatif
            optimizer = Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["acc", "mean_absolute_error"],
            )

            # On enntraîne le modèle
            training_history = model.fit(
                train_generator,
                epochs=1,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                validation_data=valid_generator,
                validation_steps=valid_generator.samples // valid_generator.batch_size,
                callbacks=[reduce_learning_rate, early_stopping],
                verbose=1,
            )

            logging.info("Entraînement terminé !")
            # On évalue le modèle sur le set de test
            test_loss, test_accuracy, test_mae = model.evaluate(test_generator)

            logging.info(f"Précision sur test: {test_accuracy}")
            logging.info(
                f"Précision finale sur validation: {training_history.history['val_acc'][-1]}"
            )

            # On sauvegarde le modèle au format h5
            model_save_path = "saved_model.h5"
            model.save(model_save_path)
            mlflow.log_artifact(model_save_path, artifact_path="model")
            os.remove(model_save_path)
            logging.info("Modèle enregistré avec succès !")

            # On génère et sauvegarde la matrice de confusion pour plus tard
            generate_confusion_matrix(test_generator, model)

            # On termine le run MLFlow
            mlflow.end_run()

            alert_system.send_alert(
                subject="Entraînement terminé avec succès !",
                message="""L'entraînement s'est terminé avec succès !
                Vous pouvez voir les résultats avec la route /results dans l'api administrateur.
                """,
            )

            # On indique que le container n'est plus actif
            with open(state_path, "w") as file:
                file.write("0")

    except Exception as e:
        logging.error(f"Un problème est survenu lors de l'entraînement : {e}")
        alert_system.send_alert(
            subject="Erreur lors de l'entraînement",
            message=f"Un problème est survenu lors de l'entraînement : {e}",
        )


# ----------------------------------------------------------------------------------------- #


@app.get("/")
def read_root():
    return {"Status": "OK"}


@app.get("/train")
async def train(background_tasks: BackgroundTasks):
    try:
        # On récupère les états des containers
        with open(preprocessing_state_path, "r") as preprocessing_file:
            preprocessing_state = preprocessing_file.read()

        with open(drift_monitor_state_path, "r") as drift_monitor_file:
            drift_monitor_state = drift_monitor_file.read()

        with open(state_path, "r") as state_file:
            state = state_file.read()

        # On vérifie que le preprocessing, drift_monitoring ou en entraînement n'est pas en cours
        if (
            preprocessing_state == "0"
            and drift_monitor_state == "0"
            and state == "0"
            and len(os.listdir(dataset_folder)) > 1
        ):
            # On lance la tâche en arrière-plan pour immédiatement retourner une réponse
            background_tasks.add_task(train_model)
            return "Entraînement du modèle lancé, merci d'attendre le mail indiquant le succès de la tâche."
        else:
            return "Un preprocessing, révision de drift ou un entraînement est en cours, merci de revenir plus tard."

    except Exception as e:
        logging.error(f"Un problème est survenu lors de l''entraînement : {e}")
        alert_system.send_alert(
            subject="Erreur lors de l'entraînement",
            message=f"Un problème est survenu lors de l''entraînement : {e}",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Un problème est survenu lors de l''entraînement : {e}",
        )


@app.get("/results")
async def results():
    """
    Renvoie les métriques du dernier modèle et de celui en production
    """
    try:

        # On récupère le run id du dernier modèle entraîné
        runs = client.search_runs(experiment_id)
        latest_run = runs[0]
        latest_run_id = latest_run.info.run_id

        # On récupère le run id du modèle actuellement utilisé pour l'inférence
        with open(os.path.join(mlruns_path, "prod_model_id.txt"), "r") as file:
            main_model_run_id = file.read()

        # On récupère les métriques et les pires scores f1 du dernier modèle entrainé
        latest_run_val_acc = latest_run.data.metrics.get("val_acc")
        latest_run_val_loss = latest_run.data.metrics.get("val_loss")
        latest_run_worst_f1_scores = get_worst_f1_scores(latest_run_id)

        # On récupère les métriques et les pires scores f1
        # du modèle actuellement utilisé pour l'inférence
        main_model_run = client.get_run(main_model_run_id)
        main_model_val_acc = main_model_run.data.metrics.get("val_acc")
        main_model_val_loss = main_model_run.data.metrics.get("val_loss")
        main_model_worst_f1_scores = get_worst_f1_scores(main_model_run_id)

        # On retourne toutes les informations utiles
        return {
            "latest_run_id": latest_run_id,
            "latest_run_val_accuracy": latest_run_val_acc,
            "latest_run_val_loss": latest_run_val_loss,
            "latest_run_worst_f1_scores": zip(
                latest_run_worst_f1_scores[0], latest_run_worst_f1_scores[1]
            ),
            "main_model_run_id": main_model_run_id,
            "main_model_val_accuracy": main_model_val_acc,
            "main_model_val_loss": main_model_val_loss,
            "main_model_worst_f1_scores": zip(
                main_model_worst_f1_scores[0], main_model_worst_f1_scores[1]
            ),
        }

    except Exception as e:
        logging.error(
            f"Un problème est survenu lors de l'affichage des résultats : {e}"
        )
        alert_system.send_alert(
            subject="Erreur lors de l'entraînement",
            message=f"Un problème est survenu lors de l'affichage des résultats : {e}",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Un problème est survenu lors de l'affichage des résultats : {e}",
        )
