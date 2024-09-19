import pandas as pd
import shutil
import os
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np


class SizeManager:
    """
    Cette classe `SizeManager` gère le redimensionnement et le nettoyage du dataset.
    On génère un CSV avec les métadonnées des images, supprime certaines classes et
    redimensionne les images en 224x224px.
    """
    def __init__(self, db_to_clean_path, target_size=(224, 224)):
        # On définit les chemins et la taille cible
        self.db_to_clean_path = db_to_clean_path
        self.target_size = target_size
        # Liste pour stocker les classes à supprimer plus tard
        self.classes_to_del_list = list()

    def getImagesInfos(self, imagePath):
        """
        Permet de récupérer les métadonnées d'une image
        """
        image = Image.open(imagePath)
        info_dict = {
            "Filename": image.filename,
            "Size": image.size,
            "Height": image.height,
            "Width": image.width,
            "Format": image.format,
            "Mode": image.mode,
        }
        return info_dict

    def get_one_bird_infos(self, birdName, fullSetPath, setPath, writer):
        """
        # Récupère et inscrit dans le CSV les infos pour un oiseau
        """
        bird_path = os.path.join(fullSetPath, birdName)
        # On supprime les espaces et on corrige le chemin
        new_bird_path = " ".join(bird_path.split())
        shutil.move(bird_path, new_bird_path)
        bird_path = new_bird_path
        # On liste les images de l'oiseau
        birdImagesList = os.listdir(bird_path)
        for file in birdImagesList:
            # On supprime les espaces
            birdName = " ".join(birdName.split())
            # On récupère les infos de l'image
            infos = self.getImagesInfos(os.path.join(bird_path, file))
            # On écrit ces infos dans le CSV
            writer.writerow(
                [
                    setPath,
                    birdName,
                    file,
                    infos["Size"],
                    infos["Height"],
                    infos["Width"],
                    infos["Format"],
                    infos["Mode"],
                ]
            )

    def generate_metadata_csv(self, filename):
        """
        Génère un CSV avec les métadonnées des images dans les différents sets (train, test, valid)
        """
        print("Génération du csv de référence")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            # On ajoute les colonnes au CSV
            writer.writerow(["set", "birdName", "filename", "size", "height", "width", "format", "mode"])
            # On parcourt les différents dossiers du dataset
            for setPath in os.listdir(self.db_to_clean_path):
                fullSetPath = os.path.join(self.db_to_clean_path, setPath)
                # Si c'est un fichier, on passe
                if os.path.isfile(fullSetPath):
                    continue
                if os.path.isfile(fullSetPath):
                    os.remove(fullSetPath)
                for birdName in tqdm(os.listdir(fullSetPath), "Generation csv : " + fullSetPath):
                    # Pour chaque oiseau dans le dataset, on récupère les infos des images
                    self.get_one_bird_infos(birdName, fullSetPath, setPath, writer)
        return filename

    def get_df_csv(self):
        """
        On charge le CSV si déjà existant, sinon on le génère
        """
        filename = self.db_to_clean_path + ".csv"
        if not os.path.isfile(filename):
            return self.generate_metadata_csv(filename)
        return filename

    def check_images_size(self, df):
        """
        Vérifie si les images sont à la bonne taille et identifie les classes à supprimer
        """
        # On filtre les images qui n'ont pas la taille souhaitée
        df_to_resize = df[df["size"] != str(self.target_size)]
        # On passe sur chaque oiseau
        for birdName in df_to_resize["birdName"].unique():
            df_birdName = df_to_resize[df_to_resize["birdName"] == birdName]
            # Ajout d'une colonne avec le ratio True or False pour savoir
            # si le ratio est proche de 1:1
            df_birdName["ratio_size"] = np.abs(df_birdName["height"] / df_birdName["width"])
            df_birdName["ratio_size_close_to_1"] = 1 - df_birdName["ratio_size"] < 0.2

            # On compte combien d'images sont proches d'un ratio 1:1
            nb_true = df_birdName['ratio_size_close_to_1'].values.sum()
            nb_false = (~df_birdName['ratio_size_close_to_1']).values.sum()
            total_images = nb_false + nb_true

            # Si moins de 80% des images ont un bon ratio, on ajoute cette classe à supprimer
            if nb_true / total_images < 0.8:
                print("Classe à supprimer : ", birdName)
                self.classes_to_del_list.append(birdName)
            else:
                # Sinon, on redimensionnera les images
                print("Classe à redimensionner : ", birdName)

    def resize_images(self, df):
        """
        Redimensionne les images qui ne sont pas à la bonne taille
        """

        print("Début du redimensionnement vers la dimension : ", str(self.target_size))
        # On filtre les images qui ne sont pas 224x224
        df_to_resize = df[df["size"] != str((224, 224))]
        count = 0
        # On passe sur chaque image à redimensionner
        for set_name, birdname, filename in zip(
            df_to_resize["set"], df_to_resize["birdName"], df_to_resize["filename"]
        ):

            img_path = os.path.join(self.db_to_clean_path, set_name, birdname, filename)
            # Si le fichier n'existe pas, on passe
            if not os.path.isfile(img_path):
                continue
            # On ouvre l'image
            img = Image.open(img_path)
            # On la redimensionne
            img_resize = img.resize(self.target_size)
            # On sauvegarde l'image redimensionnée
            img_resize_path = os.path.join(self.db_to_clean_path, set_name, birdname, filename)
            img_resize.save(img_resize_path)
        print("Image(s) redimensionnée(s) : ", count)

    def del_classes(self, df):
        """
        Supprime les classes d'oiseaux non exploitables
        """

        print("Début de la suppression des classes non exploitables")
        # On vérifie d'abord les tailles des images
        self.check_images_size(df)
        # On filtre les classes à supprimer
        df_to_delete = df[df["birdName"].isin(self.classes_to_del_list)]
        # On parcours les classes à supprimer
        for dir in os.listdir(self.db_to_clean_path):
            for birdName in df_to_delete["birdName"].unique():
                pathToDel = os.path.join(self.db_to_clean_path, dir, birdName)
                print("Suppression : ", pathToDel)
                # Si c'est un dossier, on supprime
                if os.path.isdir(pathToDel):
                    shutil.rmtree(pathToDel)

    def manage(self):
        """
        Fonction principale qui gère tout le processus
        """
        # On charge le CSV
        df = pd.read_csv(os.path.join(self.get_df_csv()))
        # On commence par la suppression des classes
        self.del_classes(df)
        # On redimensionne ensuite les images
        self.resize_images(df)
