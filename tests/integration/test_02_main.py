import os
import sys
import unittest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # Ajoutez le chemin du projet au PYTHONPATH
from app.utils.logger import setup_logger
from app.main import app

# Configuration du logger
logger = setup_logger("test_main", "test_main.log")


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info("Début de test unitaire de main.py")
        cls.client = TestClient(app)
        load_dotenv()  # Charger les variables d'environnement
        cls.API_KEY = os.getenv("API_KEY")
        cls.API_USERNAME = os.getenv("API_USERNAME")
        cls.ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
        cls.Token = ""

    def test_01_login(self):
        """
        Test de l'endpoint d'API "/token"
        Résultat attendu : Code de statut 200 et en retour un string "acess_token" et "token_type" = "bearer"
        """
        logger.info("Test 01 : login")

        data = {"username": self.API_USERNAME, "password": self.ADMIN_PASSWORD}

        response = self.client.post("/token", data=data)

        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIsInstance(json_response["access_token"], str)
        self.assertEqual(json_response["token_type"], "bearer")
        self.__class__.Token = json_response["access_token"]

    def test_02_get_status(self):
        """
        Test concluant de l'endpoint d'API "/"
        Résultat attendu : Code de statut 200
        """
        logger.info("Test 02 : get_status")

        headers = {"api-key": self.API_KEY, "Authorization": f"Bearer {self.Token}"}

        # Exécutez la requête GET avec les en-têtes spécifiés
        response = self.client.get("/", headers=headers)

        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["message"], "Bienvenue sur l'API de reconnaissance d'oiseaux")
        self.assertEqual(json_response["user"], self.API_USERNAME)

    def test_03_invalid_api_key(self):
        """
        Test de l'endpoint d'API "/" avec clé api erronée
        Résultat attendu : Code de statut 403 et en retour "detail" = "Invalid API Key"
        """
        logger.info("Test 03 : invalid_api_key")

        invalid_headers = {
            "api-key": "invalid_key",
            "Authorization": f"Bearer {self.Token}",
        }

        response = self.client.get("/", headers=invalid_headers)

        self.assertEqual(response.status_code, 403)
        json_response = response.json()
        self.assertEqual(json_response["detail"], "Invalid API Key")

    def test_04_invalid_token(self):
        """
        Test de l'endpoint d'API "/" avec token erroné
        Résultat attendu : Code de statut 401 et en retour "detail" = "Could not validate credentials"
        """
        logger.info("Test 04 : invalid_token")

        invalid_headers = {
            "api-key": self.API_KEY,
            "Authorization": "Bearer invalid_token",
        }

        response = self.client.get("/", headers=invalid_headers)

        self.assertEqual(response.status_code, 401)
        json_response = response.json()
        self.assertEqual(json_response["detail"], "Could not validate credentials")

    def test_05_invalid_login(self):
        """
        Test de l'endpoint d'API "/token" avec des identifiants invalides
        Résultat attendu : Code de statut 401 et en retour "detail" = "Incorrect username or password"
        """
        logger.info("Test 05 : invalid_login")

        data = {"username": "invalid", "password": "invalid"}

        response = self.client.post("/token", data=data)

        self.assertEqual(response.status_code, 401)
        json_response = response.json()
        self.assertEqual(json_response["detail"], "Incorrect username or password")

    # def test_06_predict(self):
    #     """
    #     Test de l'endpoint d'API "/predict"
    #     Résultat attendu : Code de statut 200 et en retour le bon label et un score entre 0 et 1
    #     """
    #     logger.info(f"Test 06 : predict")

    #     expected_label = "Toucan" # Le label attendu correspond au nom du répertoire contenu l'image envoyée
    #     image_file = os.listdir(f"./data/valid/{expected_label}")[0] # On choisit une image à tester
    #     image_path = f"./data/valid/{expected_label}/{image_file}" # On récupère le chemin de l'image

    #     with open(image_path, "rb") as image_file:

    #         files = {
    #             "file": (f"{expected_label}_image.jpg", image_file, "image/jpg")
    #         }
    #         headers = {
    #             "api-key": self.API_KEY,
    #             "Authorization": f"Bearer {self.Token}"
    #         }

    #         response = self.client.post("/predict", files=files, headers=headers)

    #         self.assertEqual(response.status_code, 200)
    #         response_json = response.json()
    #         predicted_label = str(response_json["prediction"]).capitalize()
    #         score = response_json["score"]
    #         self.assertEqual(predicted_label, expected_label, f"Expected {expected_label} but got {predicted_label}")
    #         self.assertTrue(0 <= score <= 1, f"Expected score between 0 and 1 but got {score}")

    def test_07_add_image(self):
        """
        Test de l'endpoint d'API "/add_image"
        Résultat attendu : Code de statut 200 et en retour "status" = "Image ajoutée avec succès"
        """
        logger.info("Test 07 : add_image")

        species = "Iiwi"
        image_file = os.listdir(f"./data/valid/{species}")[0]  # On choisit une image à tester
        image_path = f"./data/valid/{species}/{image_file}"  # On récupère le chemin de l'image

        with open(image_path, "rb") as image_file:

            species = "Iiwi"
            data = {
                "species": species,
                "is_new_species": "false",
                "is_unknown": "false",
            }
            files = {
                "file": ("test_image.jpg", image_file, "image/jpg"),
            }
            headers = {
                "api-key": self.API_KEY,
                "Authorization": f"Bearer {self.Token}",
            }

            response = self.client.post("/add_image", files=files, headers=headers, data=data)

            self.assertEqual(response.status_code, 200)
            json_response = response.json()
            self.assertEqual(json_response["status"], f"Image added to existing species '{species}'")

    def test_08_get_species(self):
        """
        Test de l'endpoint d'API "/get_species"
        Résultat attendu : Code de statut 200 et en retour la liste des espèces d'oiseaux
        """
        logger.info("Test 08 : get_species")

        headers = {"api-key": self.API_KEY, "Authorization": f"Bearer {self.Token}"}

        response = self.client.get("/get_species", headers=headers)

        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("species", json_response)
        self.assertTrue(len(json_response["species"]) >= 11194)

    def test_09_get_class_image(self):
        """
        Test de l'endpoint d'API "/get_class_image"
        Résultat attendu : Code de statut 200 et en retour une image
        """
        logger.info("Test 09 : get_class_image")

        species = "Sand martin"
        headers = {"api-key": self.API_KEY, "Authorization": f"Bearer {self.Token}"}
        params = {"classe": species}

        response = self.client.get("/get_class_image", params=params, headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(
            response.headers["content-disposition"].lower().replace("%20", " "),
            f"attachment; filename*=utf-8''{species.lower()}_image.jpg",
        )

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        logger.info("Fin de test unitaire de main.py")


if __name__ == "__main__":
    unittest.main()
