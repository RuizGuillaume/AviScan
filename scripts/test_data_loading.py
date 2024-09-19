import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_manager import DataManager
from app.utils.logger import setup_logger

logger = setup_logger("test_data_loading", "test_data_loading.log")


def test_load_new_data():
    data_manager = DataManager()
    new_data = data_manager.load_new_data()

    logger.info(f"Nombre total d'images chargées : {len(new_data)}")

    # Afficher les 10 premières entrées pour vérification
    for i, (image_path, true_class) in enumerate(new_data[:10]):
        logger.info(f"Image {i+1}: {image_path}, Classe: {true_class}")

    # Vérifier que toutes les entrées ont une vraie classe
    all_have_true_class = all(true_class is not None for _, true_class in new_data)
    logger.info(f"Toutes les entrées ont une vraie classe : {all_have_true_class}")


if __name__ == "__main__":
    test_load_new_data()
