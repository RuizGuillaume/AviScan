import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # Ajoutez le chemin du projet au PYTHONPATH
import unittest
import datetime
from app.utils.logger import setup_logger
from training.train_model import train_model

# Configuration du logger
logger = setup_logger("test_train_model", "test_train_model.log")


class TestTrainModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info("Début de test unitaire de train_model.py")
        cls.models_folder = "./models"

    def test_01_main(self):
        """
        Test de l'entrainement du modèle
        Résultat attendu : Création du fichier birds_model_XXXXXXXXXXXX.h5
        """
        logger.info("Test 01 : main")

        before_train_time = datetime.datetime.now()  # Date et heure avant l'étape de preprocessing

        train_model()  # Entrainement du modèle

        # Trouver le sous-dossier le plus récent basé sur la date de modification
        subfolders = [f.path for f in os.scandir(self.models_folder) if f.is_dir()]
        most_recent_folder = max(subfolders, key=os.path.getmtime)

        model_path = f"{most_recent_folder}/saved_model.pb"  # Chemin du dernier modèle

        self.assertTrue(os.path.isfile(f"{model_path}"), f"Le modèle {model_path} n'existe pas.")

        new_model_creation_time = datetime.datetime.fromtimestamp(
            os.path.getctime(model_path)
        )  # Date et heure de création du nouveau modèle
        self.assertTrue(
            before_train_time <= new_model_creation_time,
            f"Le dernier modèle {model_path} date d'avant cette étape d'entrainement.",
        )

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        logger.info("Fin de test unitaire de train_model.py")


if __name__ == "__main__":
    unittest.main()
