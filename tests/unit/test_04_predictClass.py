import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # Ajoutez le chemin du projet au PYTHONPATH
import unittest
from app.utils.logger import setup_logger
from app.models.predictClass import predictClass

# Configuration du logger
logger = setup_logger("predictClass", "predictClass.log")


class testSingleInferenceImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info("Début de test unitaire de predictClass.py")
        test_path = "./data/test"
        cls.predict_class = predictClass(model_path=None, test_path=test_path, img_size=(224, 224))

    def test_01_main(self):
        """
        Test d'inférence du modèle
        Résultat attendu : Une prédiction composée d'un nom d'espèce d'oiseau, et un score entre 0 et 1
        """
        logger.info("Test 01 : main")

        expected_label = "Crimson sunbird"
        image_file = os.listdir(f"./data/valid/{expected_label}")[0]  # On choisit une image à tester
        image_path = f"./data/valid/{expected_label}/{image_file}"  # On récupère le chemin de l'image

        class_name, confidence = self.predict_class.predict(image_path)  # Prédiction

        self.assertTrue(bool(class_name.strip()), "La classe prédite par le modèle est vide.")
        self.assertTrue(
            0 <= confidence <= 1,
            "Le score de la prédiction n'est pas compris dans l'intervalle [0;1].",
        )

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        logger.info("Fin de test unitaire de predictClass.py")


if __name__ == "__main__":
    unittest.main()
