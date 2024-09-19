import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
import shutil
import json
import glob
from app.utils.logger import setup_logger
from dotenv import load_dotenv
load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi


logger = setup_logger("download_dateset", "download_dateset.log")


def download_dataset(dataset_name: str = "gpiosenka/100-bird-species", destination_folder: str = "./data"):
    """
    Télécharger la base de données d'oiseaux sur Kaggle et le placer dans le dossier passé en paramètre
    """
    # Initialisation de l'API de Kaggle
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    # endregion

    # region Vérification des conditions pour télécharger le dataset
    # Récupérer la date de la dernière mise à jour du dataset Kaggle
    datasets = kaggle_api.dataset_list(
        search=dataset_name, sort_by="hottest"
    )  # Renvoie la liste des datasets qui conrrepondent aux critères de recherche
    online_dataset_version = datetime(1970, 1, 1)  # On initialise la date du dataset Kaggle
    for dataset in datasets:
        if dataset.ref == dataset_name:  # Pour le bon dataset
            online_dataset_version = dataset.lastUpdated  # On récupère la version du dataset sur Kaggle
            dataset_info = {  # On sauvegarde les informations pour créer notre fichier dataset_version.json
                "dataset_name": dataset.ref,
                "last_updated": dataset.lastUpdated.strftime("%Y-%m-%d %H:%M:%S"),
            }
    # Si on a un fichier dataset_version.json et que notre dataset est à jour, on ne retélécharge pas le dataset
    if os.path.isfile(f"{destination_folder}/dataset_version.json"):
        with open(f"{destination_folder}/dataset_version.json", "r") as file:
            current_dataset_version = datetime.strptime(
                json.load(file).get("last_updated"), "%Y-%m-%d %H:%M:%S"
            )  # On récupère la version de notre dataset
            if current_dataset_version >= online_dataset_version:
                logger.info("Le dataset est à jour, on ne le télécharge pas.")
                return
            else:
                logger.info("La version du dataset nécéssite une mise à jour.")
    # endregion

    # regionSupressions des dossiers/fichiers à remplacer
    # Dossiers
    folders = ["train", "test", "valid"]
    for folder in folders:
        folder_to_delete = f"{destination_folder}/{folder}"
        if os.path.exists(folder_to_delete) and os.path.isdir(folder_to_delete):
            shutil.rmtree(folder_to_delete)

    # Fichiers
    files = ["birds.csv", "birds_backup.csv"]
    for file in files:
        file_to_delete = f"{destination_folder}/{file}"
        if os.path.exists(file_to_delete) and os.path.isfile(file_to_delete):
            os.remove(file_to_delete)
    # endregion

    # On télécharge le fichier
    kaggle_api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)

    # On enregistre les informations du dataset Kaggle dans le fichier dataset_version.json
    if dataset_info:
        with open(f"{destination_folder}/dataset_version.json", "w") as file:
            json.dump(dataset_info, file, indent=4)

    logger.info("Dataset téléchargé et extrait !")

    # region suppression des fichiers inutiles
    files_to_delete = glob.glob(
        os.path.join(destination_folder, "EfficientNetB0*.h5")
    )  # Fichiers correspondant au motif
    for file_path in files_to_delete:  # On supprime les fichiers trouvés
        if os.path.isfile(file_path):
            os.remove(file_path)
    # endregion


if __name__ == "__main__":
    download_dataset()
