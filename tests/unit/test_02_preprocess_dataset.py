import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # Ajoutez le chemin du projet au PYTHONPATH
import unittest
from PIL import Image
from app.utils.logger import setup_logger
from preprocessing.preprocess_dataset import CleanDB


# Configuration du logger
logger = setup_logger("test_preprocess_dataset", "test_preprocess_dataset.log")


class TestPreprocessDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info("Début de test unitaire de preprocess_dataset.py")
        cls.species = "Crimson sunbird"
        cls.data_folder = "./data"
        cls.species_raw_folder = (
            f"{cls.data_folder}/train/{cls.species.upper()}"  # Avant preprocessing, les dossiers sont en majuscules
        )
        # Après preprocessing, les dossiers sont en minuscules (sauf la premiere lettre)
        cls.species_processed_folder = f"{cls.data_folder}/train/{cls.species.capitalize()}"

    def test_01_main(self):
        """
        Test du preprocessing du dataset
        Résultat attendu : Données nettoyées dans le dossier data
        """
        logger.info("Test 01 : main")

        clean_db = CleanDB(self.data_folder, treshold=False, test_mode=True)

        # Le dossier 'CRIMSON BIRD' doit exister, mais pas 'Crimson sunbird'
        self.assertTrue(
            "CRIMSON SUNBIRD" in os.listdir(f"{self.data_folder}/train"),
            f"Avant preprocessing, le répertoire {self.species_raw_folder} en majuscule devrait exister.",
        )
        self.assertFalse(
            "Crimson sunbird" in os.listdir(f"{self.data_folder}/train"),
            f"Avant preprocessing, le répertoire {self.species_processed_folder} en minuscule ne devrait pas exister.",
        )

        clean_db.cleanAll()  # Étape de preprocessing

        # Le dossier 'CRIMSON BIRD' ne doit plus exister, mais 'Crimson sunbird' oui
        self.assertFalse(
            "CRIMSON SUNBIRD" in os.listdir(f"{self.data_folder}/train"),
            f"Après preprocessing, le répertoire {self.species_raw_folder} en majuscule ne devrait pas exister.",
        )
        self.assertTrue(
            "Crimson sunbird" in os.listdir(f"{self.data_folder}/train"),
            f"Après preprocessing, le répertoire {self.species_processed_folder} en minuscule devrait exister.",
        )

        # Test de l'existence du fichier birds_list.csv
        self.assertTrue(
            os.path.isfile(f"{self.data_folder}/birds_list.csv"),
            f"Le fichier {self.data_folder}/birds_list.csv n'existe pas.",
        )

        # Test de la résolution des images
        image_file = os.listdir(f"{self.data_folder}/test/{self.species}")[0]  # On choisit une image à tester
        image_path = f"{self.data_folder}/test/{self.species}/{image_file}"  # On récupère le chemin de l'image

        self.assertTrue(os.path.isfile(f"{image_path}"), f"Le fichier {image_path} n'existe pas.")
        with Image.open(f"{image_path}") as img:
            self.assertTrue(
                img.size == (224, 224),
                f"L'image {image_path} ne possède pas les dimensions attendues : {img.size} au lieu de (224, 224).",
            )

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        logger.info("Fin de test unitaire de preprocess_dataset.py")


if __name__ == "__main__":
    unittest.main()
