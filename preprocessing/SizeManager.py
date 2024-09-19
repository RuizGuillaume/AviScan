import pandas as pd
import shutil
import os
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np


class SizeManager:
    def __init__(self, db_to_clean_path, target_size=(224, 224)):
        self.db_to_clean_path = db_to_clean_path
        self.target_size = target_size
        self.classes_to_del_list = list()

    def getImagesInfos(self, imagePath):
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
        # Recupere et inscrit dans le csv les infos pour un oiseau (une classe)
        bird_path = os.path.join(fullSetPath, birdName)
        new_bird_path = " ".join(bird_path.split())
        shutil.move(bird_path, new_bird_path)
        bird_path = new_bird_path
        birdImagesList = os.listdir(bird_path)
        for file in birdImagesList:
            birdName = " ".join(birdName.split())
            try:
                infos = self.getImagesInfos(os.path.join(bird_path, file))
            except Exception:
                print("zbra")
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
        # Fonction générant un csv qui présente les metadatas de trois set
        print("Génération du csv de référence")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(
                [
                    "set",
                    "birdName",
                    "filename",
                    "size",
                    "height",
                    "width",
                    "format",
                    "mode",
                ]
            )
            for setPath in os.listdir(self.db_to_clean_path):
                fullSetPath = os.path.join(self.db_to_clean_path, setPath)
                if os.path.isfile(fullSetPath):
                    # ce n'est pas un dossier
                    continue
                if os.path.isfile(fullSetPath):
                    os.remove(fullSetPath)
                for birdName in tqdm(os.listdir(fullSetPath), "Generation csv : " + fullSetPath):
                    self.get_one_bird_infos(birdName, fullSetPath, setPath, writer)
        return filename

    def get_df_csv(self):
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
        # On passe un df avec toute les images du dataset et une taille cible
        # Toutes les images qui n'ont pas la taille cible sont redimmensionnées
        # ATTENTION : A faire après avoir réaliser les éventuel suppression de classe
        print("Début du redimensionnement vers la dimension : ", str(self.target_size))
        df_to_resize = df[df["size"] != str((224, 224))]
        count = 0
        for set_name, birdname, filename in zip(
            df_to_resize["set"], df_to_resize["birdName"], df_to_resize["filename"]
        ):

            img_path = os.path.join(self.db_to_clean_path, set_name, birdname, filename)
            if not os.path.isfile(img_path):
                # Le fichier a été supprimé
                continue
            img = Image.open(img_path)
            img_resize = img.resize(self.target_size)
            img_resize_path = os.path.join(self.db_to_clean_path, set_name, birdname, filename)
            img_resize.save(img_resize_path)
        print("Image(s) redimensionnée(s) : ", count)

    def del_classes(self, df):
        print("Début de la suppression des classes non exploitables")
        # Suppression de toute les classes spécifiée dans le df
        self.check_images_size(df)
        df_to_delete = df[df["birdName"].isin(self.classes_to_del_list)]
        for dir in os.listdir(self.db_to_clean_path):
            for birdName in df_to_delete["birdName"].unique():
                pathToDel = os.path.join(self.db_to_clean_path, dir, birdName)
                print("Suppression : ", pathToDel)
                if os.path.isdir(pathToDel):
                    shutil.rmtree(pathToDel)

    def manage(self):
        df = pd.read_csv(os.path.join(self.get_df_csv()))
        # print("Début du nettoyage")
        self.del_classes(df)
        self.resize_images(df)
