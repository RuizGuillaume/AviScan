import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # Ajoutez le chemin du projet au PYTHONPATH
import unittest
from app.utils.logger import setup_logger
from scripts.downloadDataset import download_dataset

# Configuration du logger
logger = setup_logger("test_DownloadDataset", "test_DownloadDataset.log")


class TestDownloadDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info("Début de test unitaire de downloadDataset.py")

        cls.dataset_name = "gpiosenka/100-bird-species"

        cls.main_folder = "./data"
        cls.train_folder = f"{cls.main_folder}/train"
        cls.test_folder = f"{cls.main_folder}/test"
        cls.valid_folder = f"{cls.main_folder}/valid"
        cls.species_folder = f"{cls.main_folder}/test/ZEBRA DOVE"
        cls.dataset_version = f"{cls.main_folder}/dataset_version.json"

    def test_01_main(self):
        """
        Test du téléchargement du dataset Kaggle BIRDS xxx SPECIES
        Résultat attendu : Présence d'un dossier "train", d'un dossier "test", d'un dossier "valid",
        d'un fichier "birds.csv" et d'un fichier "dataset_version.json"
        """
        logger.info("Test 01 : main")

        # Téléchargement du dataset Kaggle
        download_dataset(dataset_name=self.dataset_name, destination_folder=self.main_folder)

        self.assertTrue(
            os.path.exists(self.main_folder),
            f"Le répertoire {self.main_folder} n'existe pas.",
        )  # Existence du dossier data
        self.assertTrue(
            os.path.exists(self.train_folder),
            f"Le répertoire {self.train_folder} n'existe pas.",
        )  # Existence du dossier train
        self.assertTrue(
            os.path.exists(self.test_folder),
            f"Le répertoire {self.test_folder} n'existe pas.",
        )  # Existence du dossier test
        self.assertTrue(
            os.path.exists(self.valid_folder),
            f"Le répertoire {self.valid_folder} n'existe pas.",
        )  # Existence du dossier valid
        self.assertTrue(
            os.path.exists(self.species_folder),
            f"Le répertoire {self.species_folder} n'existe pas.",
        )  # Existence du dossier d'une espèce
        self.assertTrue(
            os.listdir(self.species_folder),
            f"Le répertoire {self.species_folder} est vide après le téléchargement.",
        )  # Dossier d'une espèce non vide
        self.assertTrue(
            os.path.isfile(self.dataset_version),
            f"Le fichier {self.dataset_version} n'existe pas.",
        )  # Existence du fichier de version du dataset

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        logger.info("Fin de test unitaire de downloadDataset.py")


if __name__ == "__main__":
    unittest.main()
