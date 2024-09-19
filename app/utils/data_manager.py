import os
from app.utils.logger import setup_logger

logger = setup_logger("data_manager", "logs/data_manager.log")


class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.valid_dir = os.path.join(data_dir, "valid")
        self.test_dir = os.path.join(data_dir, "test")

    def get_class_names(self):
        logger.info("Récupération des noms de classes")
        class_names = sorted([d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))])
        logger.info(f"Nombre de classes trouvées : {len(class_names)}")
        return class_names

    def load_new_data(self):
        logger.info("Chargement des nouvelles données")
        image_paths = []
        for root, _, files in os.walk(self.test_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(root, file)
                    true_class = os.path.basename(os.path.dirname(image_path))
                    image_paths.append((image_path, true_class))
                    logger.info(f"Nouvelle image trouvée : {image_path}")
        logger.info(f"Total de {len(image_paths)} nouvelles images trouvées")
        return image_paths

    def get_existing_classes(self):
        logger.info("Récupération des classes existantes")
        classes = self.get_class_names()
        logger.info(f"Classes existantes : {classes}")
        return set(classes)

    def move_to_train(self, image_path, class_name):
        logger.info(f"Déplacement de l'image {image_path} vers la classe {class_name}")
        dest_dir = os.path.join(self.train_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(image_path))
        os.rename(image_path, dest_path)
        logger.info(f"Image déplacée avec succès vers {dest_path}")

    def create_new_class(self, class_name):
        logger.info(f"Création d'une nouvelle classe : {class_name}")
        new_class_path = os.path.join(self.train_dir, class_name)
        os.makedirs(new_class_path, exist_ok=True)
        logger.info(f"Nouvelle classe créée : {new_class_path}")

    def remove_from_test(self, image_path):
        logger.info(f"Suppression de l'image du dossier de test : {image_path}")
        os.remove(image_path)
        logger.info(f"Image supprimée : {image_path}")

    def get_class_distribution(self):
        logger.info("Calcul de la distribution des classes")
        distribution = {}
        for class_name in self.get_class_names():
            class_path = os.path.join(self.train_dir, class_name)
            num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            distribution[class_name] = num_images
        logger.info(f"Distribution des classes calculée : {distribution}")
        return distribution

    def get_total_images(self):
        logger.info("Comptage du nombre total d'images")
        total = sum(self.get_class_distribution().values())
        logger.info(f"Nombre total d'images : {total}")
        return total
