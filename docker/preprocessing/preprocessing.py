import os
import shutil
import time
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime
import json
import schedule
from CleanDB import CleanDB
from DatasetCorrection import DatasetCorrection
from alert_system import AlertSystem

# On créer les différents chemins
volume_path = "volume_data"
dataset_raw_path = os.path.join(volume_path, "dataset_raw")
dataset_clean_path = os.path.join(volume_path, "dataset_clean")
dataset_version_path = os.path.join(dataset_raw_path, "dataset_version.json")
classes_tracking_path = os.path.join(dataset_raw_path, "classes_tracking.json")
state_folder = os.path.join(volume_path, "containers_state")
state_path = os.path.join(state_folder, "preprocessing_state.txt")
training_state_path = os.path.join(state_folder, "training_state.txt")
monitoring_state_path = os.path.join(state_folder, "drift_monitor_state.txt")
log_folder = os.path.join(volume_path, "logs")

# On créer les dossiers si nécessaire
os.makedirs(log_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)
os.makedirs(dataset_raw_path, exist_ok=True)
os.makedirs(dataset_clean_path, exist_ok=True)

# On indique l'état de départ du container, donc 0 = inactif
with open(state_path, "w") as file:
    file.write("0")

# On configure le logging pour les informations et les erreurs
logging.basicConfig(
    filename=os.path.join(log_folder, "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S %p",
)

# On instancie la classe qui permet d'envoyer des alertes par email
alert_system = AlertSystem()


def save_json(filepath, dict):
    """
    Permet d'enregistrer un dictionnaire en format .json
    """
    with open(filepath, "w") as file:
        json.dump(dict, file, indent=4)


def get_new_classes(updated_dict):
    """
    Premet de récupérer en permanence les nouvelles classes.
    """
    # On parcours les classes dans train et on déclare une liste vide des nouvelles classs
    updated_list = os.listdir(os.path.join(dataset_raw_path, "train"))
    new_classes_to_track = []

    # On ajoute à la liste toutes les classes présentes dans train qui ne sont pas dans base_dict
    for classe in updated_list:
        if classe not in updated_dict["originals"] and classe not in updated_dict["new"]:
            new_classes_to_track.append(classe)
    return new_classes_to_track


def start_cleaning(new_classes_to_track=[]):
    """
    Lance le nettoyage de la base de données en copiant les classes suffisamenent grandes de
    dataset_raw vers dataset_clean et en appliquant le preprocessing.
    """
    # La fonction reste en attente tant que les containers de training et drift_monitoring sont actifs
    with open(training_state_path, "r") as training_file:
        with open(monitoring_state_path, "r") as monitoring_file:
            training_state = training_file.read()
            monitoring_state = monitoring_file.read()
            while training_state == "1" or monitoring_state == "1":
                time.sleep(5)
                training_state = training_file.read()
                monitoring_state = monitoring_file.read()

    # On indique que ce container est actif
    with open(state_path, "w") as file:
        file.write("1")

    # On instancie la classe qui s'occupe de tout le nettoyage (code créé dans un autre projet)
    cleanDB = CleanDB(dataset_clean_path, treshold=False)

    # On supprime ce qui est présent dans dataset_clean et on copie le contenu brut
    shutil.rmtree(dataset_clean_path)
    os.makedirs(dataset_clean_path)
    time.sleep(5)
    shutil.copytree(dataset_raw_path, dataset_clean_path, dirs_exist_ok=True)

    # On supprime tous les fichiers qui sont en double avec dataset_raw
    if os.path.exists(os.path.join(dataset_clean_path, "dataset_version.json")):
        os.remove(os.path.join(dataset_clean_path, "dataset_version.json"))
    os.remove(os.path.join(dataset_clean_path, "birds.csv"))
    os.remove(os.path.join(dataset_clean_path, "birds_list.csv"))

    # On récupère les nouvelles classes et on les supprime, car elles ne doivent pas être traitées
    # tant qu'elles sont condisérées comme nouvelles (trop peu d'images)
    if new_classes_to_track:
        for classe in new_classes_to_track:
            shutil.rmtree(os.path.join(dataset_clean_path, "train", classe))

    # On lance le preprocessing
    cleanDB.cleanAll()

    # On indique que le container n'est plus actif
    time.sleep(5)
    with open(state_path, "w") as file:
        file.write("0")


def auto_update_dataset(dataset_name, destination, first_launch=False):
    """
    Permet de vérifier lorsqu'une nouvelle version du dataset est disponible et de le télécharger.
    """

    # Initialisation de l'API de Kaggle
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    # Renvoie la liste des datasets qui correspondent aux critères de recherche
    datasets = kaggle_api.dataset_list(search=dataset_name, sort_by="hottest")
    # On parcours la liste reçu pour trouver le nom qui corresponds à notre recherche
    for dataset in datasets:
        if dataset.ref == dataset_name:
            # On récupère la version du dataset sur Kaggle
            online_dataset_version = dataset.lastUpdated
            # On sauvegarde les informations pour créer notre fichier dataset_version.json
            dataset_info = {
                "dataset_name": dataset.ref,
                "last_updated": online_dataset_version.strftime("%Y-%m-%d %H:%M:%S"),
            }

    # Si c'est le premier lancement, on télécharge forcément le dataset
    # Sinon, on rentre dans ce code
    if first_launch is False:

        logging.info("Vérification de la version du dataset...")

        # On récupère la version du dataset local
        with open(dataset_version_path, "r") as file:
            current_dataset_version = datetime.strptime(json.load(file).get("last_updated"), "%Y-%m-%d %H:%M:%S")
            # On termine la fonction ici si aucune version plus récente n'est disponible
            if current_dataset_version >= online_dataset_version:
                logging.info("Le dataset est à jour, on ne le télécharge pas.")
                return
            # Sinon, on continue le programme
            else:
                logging.info("La version du dataset nécéssite une mise à jour.")

    # On télécharge le fichier zip du dataset via l'API de Kaggle dans un dossier temporaire
    temp_destination = os.path.join(destination, "temp")
    kaggle_api.dataset_download_files(dataset_name, path=temp_destination, unzip=True)

    # On supprime ce dont on a pas besoin
    os.remove(os.path.join(temp_destination, "EfficientNetB0-525-(224 X 224)- 98.97.h5"))

    # On instancie la classe qui se charge de corriger les noms d'éspèces afin
    # d'uniformiser avec une liste de 11 000 espèces
    datasetCorrection = DatasetCorrection(db_to_clean=temp_destination, test_mode=False)
    # On lance la correction
    datasetCorrection.full_correction()

    # Une fois la correction terminée, on ajoute les données à dataset_raw
    shutil.copytree(temp_destination, destination, dirs_exist_ok=True)
    shutil.rmtree(temp_destination)
    logging.info("Mise à jour du dataset terminée !")
    if not first_launch:
        alert_system.send_alert(
            subject="Un nouveau dataset vient d'être téléchargé !",
            message="""Un nouveau dataset vient d'être téléchargé.
                                S'il contient suffisament de nouvelles images ou des nouvelles
                                classes, le preprocessing se déclenchera automatiquement
                                et un autre mail sera envoyé.""",
        )

    # Lors du premier lancement, on rajoute ce fichier à la fin du preprocessing
    # pour permettre de relancer le processus complet en cas de crash
    # car c'est la présence de ce fichier qui indique la nécessité
    # de télécharger le dataset la première fois
    if first_launch is False:
        # On actualise la version du dataset local
        with open(dataset_version_path, "w") as file:
            json.dump(dataset_info, file, indent=4)

    else:
        # On donnes les informations pour enregistrer le fichier
        return dataset_info


def refresh_images_count(base_dict, updated_dict, dict_class_key, dict_count_key):
    """
    Permet d'actualiser le nombre d'images dans le dictionnaire pour les clés spécifiées.
    """
    updated_dict[dict_count_key] = []
    for classe in updated_dict[dict_class_key]:
        updated_dict[dict_count_key].append(len(os.listdir(os.path.join(dataset_raw_path, "train", classe))))

    return sum(updated_dict[dict_count_key]) - sum(base_dict[dict_count_key])


# ----------------------------------------------------------------------------------------- #

try:
    # On se base sur l'existence de ce fichier pour savoir si il faut télécharger le dataset
    if not os.path.exists(dataset_version_path):
        # On indique que ce container se lance pour la première fois
        with open(state_path, "w") as file:
            file.write("2")
        logging.info("Téléchargement du dataset car aucun n'est présent...")
        # On télécharge le dataset
        dataset_info = auto_update_dataset(dataset_name="gpiosenka/100-bird-species",
                                           destination=dataset_raw_path, first_launch=True)
        # On lance un premier preprocessing
        logging.info("Lancement du premier preprocessing...")
        start_cleaning()

        logging.info("Création du fichier de tracking des classes...")
        # On créer le dictionnaire qui s'occupe du suivi des classes
        classes_tracking_base = {
            # On ajoute le nom des classes présentes dans train
            "originals": os.listdir(os.path.join(dataset_raw_path, "train")),
            "originals_count": [],
            "new": [],
            "new_count": [],
        }

        # On ajoute ensuite le nombre d'images par classes
        for folder in os.listdir(os.path.join(dataset_raw_path, "train")):
            classes_tracking_base["originals_count"].append(
                len(os.listdir(os.path.join(dataset_raw_path, "train", folder)))
            )

        # On enregistre le fichier
        save_json(classes_tracking_path, classes_tracking_base)
        # On indique que ce container est inactif
        with open(state_path, "w") as file:
            file.write("0")

        # On actualise la version du dataset local
        with open(dataset_version_path, "w") as file:
            json.dump(dataset_info, file, indent=4)

        logging.info("Le dataset de base a bien été téléchargé et le preprocessing est terminé !")
        alert_system.send_alert(
            subject="Le dataset de base est prêt !",
            message="""Le dataset de base a été téléchargé et le preprocessing a été effecuté,
            vous pouvez donc maintenant utiliser l'application dans son entiereté !""",
        )
except Exception as e:
    logging.error(f"Erreur lors du premier lancement : {e}")
    alert_system.send_alert(subject="Erreur lors du preprocessing", message=f"Erreur lors du premier lancement : {e}")

try:
    # On charge le dictionnaire de tracking des classes
    with open(classes_tracking_path, "r") as file:
        classes_tracking_base = json.load(file)
        # On déclare un second dictionnaire qui lui est actualisé en permanence
        # (pour pouvoir comparer le dictionnaire actualisé avec celui de base et voir les différences)
        classes_tracking_updated = classes_tracking_base.copy()
        logging.info("Chargement des données de tracking du dataset")
except Exception as e:
    logging.error(f"Erreur lors de l'ouverture du fichier de tracking : {e}")
    alert_system.send_alert(
        subject="Erreur lors du preprocessing",
        message=f"Erreur lors de l'ouverture du fichier de tracking : {e}"
    )


# Tous les jours à 02h, on vérifie la présence d'un nouveau dataset
schedule.every().day.at("02:00").do(auto_update_dataset, "gpiosenka/100-bird-species", dataset_raw_path)

# ----------------------------------------------------------------------------------------- #

# On lance la partie qui tourne en permanence
while True:

    try:

        # On actualise le nombre d'images pour les classes originales (dataset Kaggle)
        # et les classes nouvelles (ajoutées par les utilisateurs)
        original_classes_added_images = refresh_images_count(
            classes_tracking_base, classes_tracking_updated, "originals", "originals_count"
        )
        new_classes_added_images = refresh_images_count(
            classes_tracking_base, classes_tracking_updated, "new", "new_count"
        )

        # On récupère la liste des classes non "matures"
        # (qui n'ont pas (encore) assez d'images pour passer le  preprocessing)
        new_classes_to_track = get_new_classes(classes_tracking_updated)

        # On calcule le nombre total d'images ajoutées depuis le dernier preprocessing
        total_added_images = original_classes_added_images + new_classes_added_images

        # On spécifie le seuil à partir duquel il y a suffisamenent d'images ajoutées,
        # à savoir 1% du nombre total d'images dans le dataset
        minimum_nbr_images = 0.01 * (
            sum(classes_tracking_base["originals_count"]) + sum(classes_tracking_base["new_count"])
        )

        # Si le nombre total d'images ajoutées dépasse le seuil, on lance le preprocessing,
        # on avertit par mail les administrateurs et actualise le tracking de base
        if total_added_images > minimum_nbr_images:
            logging.info(f"{total_added_images} nouvelles images ajoutées, lancement du preprocessing...")
            # On lance le preprocessing (sans les classes non "matures")
            start_cleaning(new_classes_to_track)
            # On transmets les données du dictionnaire actualisé dans le dictionnaire de base
            # car cela sera la nouvelle référence
            classes_tracking_base = classes_tracking_updated.copy()
            # On actualise le fichier
            save_json(classes_tracking_path, classes_tracking_base)
            logging.info("Preprocessing terminé avec succès !")
            alert_system.send_alert(
                subject="Nouvelles images ajoutées !",
                message=f"""{total_added_images} images viennent d'être ajoutées
                            dans le dataset, le prepreocessing a été effecuté,
                            il est donc possible d'entraîner le modèle.""",
            )
            time.sleep(5)

        # Cette variable permet de compter le nombre de nouvelles classes ajoutées
        # pour faire un seul preprocessing général
        nbr_new_classes = 0
        # On parcours la liste des classes non "matures"
        for classe in new_classes_to_track:
            # On récupère le nombre d'images de chaque classe
            nbr_images = len(os.listdir(os.path.join(dataset_raw_path, "train", classe)))
            # Si une classe à au moins le même nombre d'images que la plus petite classe du dataset Kaggle,
            # on l'ajoute définitivement
            if nbr_images >= min(classes_tracking_base["originals_count"]):
                nbr_new_classes += 1
                logging.info(f"Nouvelle classe {classe} ajoutée")
                # On ajoute la nouvelle classe ainsi que son nombre d'image dans le dictionnaire
                classes_tracking_updated["new"].append(classe)
                classes_tracking_updated["new_count"].append(
                    len(os.listdir(os.path.join(dataset_raw_path, "train", classe)))
                )

        # On actualise à nouveau les classes suivies, avec donc les classes ajoutées en moins
        new_classes_to_track = get_new_classes(classes_tracking_updated)
        if nbr_new_classes > 0:
            logging.info(f"{nbr_new_classes} classes ajoutées, lancement du preprocessing...")
            # On lance le preprocessing (toujours sans les classes non "matures")
            start_cleaning(new_classes_to_track)
            # On transmets les données du dictionnaire actualisé dans le dictionnaire de base
            # car cela sera la nouvelle référence
            classes_tracking_base = classes_tracking_updated.copy()
            # On actualise le fichier
            save_json(classes_tracking_path, classes_tracking_base)
            logging.info("Preprocessing terminé avec succès !")
            time.sleep(5)
            alert_system.send_alert(
                subject="Nouvelle(s) classe(s) ajoutée(s) !",
                message=f"""{nbr_new_classes} classes viennent
                            d'être ajoutées car elles ont suffisament d'images,
                            et le preprocessing a été effecuté.
                            Il est possible de lancer un entraînement.""",
            )
    except Exception as e:
        logging.error(f"Erreur lors du tracking des classes : {e}")
        alert_system.send_alert(
            subject="Erreur lors du preprocessing",
            message=f"Erreur lors du tracking des classes : {e}"
        )

    try:
        # On fait tourner le scheduler pour le téléchargement automatique du dataset
        schedule.run_pending()
    except Exception as e:
        logging.error(f"Error lors de la recherche de mise à jour du dataset : {e}")
        alert_system.send_alert(
            subject="Erreur lors du preprocessing",
            message=f"Error lors de la recherche de mise à jour du dataset : {e}"
        )
    # On attends 5 secondes à chaque exécution de la boucle pour ne pas saturer le processeur
    time.sleep(5)
