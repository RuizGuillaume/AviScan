import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import logging

logging.basicConfig(level=logging.INFO)


class predictClass:
    def __init__(self, model_path=None, test_path="./data/test", img_size=(224, 224)):
        self.img_size = img_size
        self.path_test = os.path.join(test_path)

        if model_path is None:
            # Chercher le modèle le plus récent dans le dossier 'models'
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(base_dir, "models")
            model_folders = [f for f in os.listdir(models_dir) if f.startswith("saved_model_")]
            if not model_folders:
                raise FileNotFoundError("Aucun modèle sauvegardé n'a été trouvé.")
            latest_model = max(
                model_folders,
                key=lambda x: os.path.getmtime(os.path.join(models_dir, x)),
            )
            self.model_path = os.path.join(models_dir, latest_model)
        else:
            self.model_path = model_path

        # Vérifier l'existence du dossier de modèle et du dossier de test
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Le dossier de modèle {self.model_path} n'existe pas.")
        if not os.path.exists(self.path_test):
            raise FileNotFoundError(f"Le dossier de test {self.path_test} n'existe pas.")

        # Configurer GPU si disponible
        self.configure_gpu()

        try:
            self.model = tf.saved_model.load(self.model_path)
            self.predict_fn = self.model.signatures["serving_default"]
            logging.info("Modèle chargé avec succès.")

            # Obtenir les noms de classes à partir du dossier de test
            self.class_names = [d for d in os.listdir(self.path_test) if os.path.isdir(os.path.join(self.path_test, d))]
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation : {str(e)}")
            raise

    def configure_gpu(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info("GPU(s) configuré(s) pour une utilisation dynamique de la mémoire.")
            except RuntimeError as e:
                logging.error(f"Erreur lors de la configuration du GPU : {e}")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"L'image {image_path} n'existe pas.")

        try:
            img = image.load_img(image_path, target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            img_ready = preprocess_input(img_array_expanded_dims)

            predictions = self.predict_fn(tf.constant(img_ready))
            output_name = list(predictions.keys())[0]
            prediction = predictions[output_name].numpy()

            highest_score_index = np.argmax(prediction)
            meilleure_classe = self.class_names[highest_score_index]
            highest_score = float(np.max(prediction))

            logging.info(f"Prédiction effectuée : classe = {meilleure_classe}, score = {highest_score}")
            return meilleure_classe, highest_score
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction : {str(e)}")
            raise

    def get_class_names(self):
        return self.class_names
