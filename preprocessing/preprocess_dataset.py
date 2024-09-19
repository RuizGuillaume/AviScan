import os
import random
import shutil
import time
from tqdm import tqdm
from .DatasetCorrection import DatasetCorrection
from .UnderSampling import UnderSamplerImages
from .SizeManager import SizeManager


class CleanDB:
    def __init__(self, db_to_clean, treshold=160, random_state=True, test_mode: bool = False):
        """
        Initialisation des chemin vers la db à nettoyer"""
        self.db_to_clean_path = db_to_clean
        self.random_state = random_state
        self.treshold = treshold
        self.test_mode = test_mode
        self.all_file_path = os.path.join(self.db_to_clean_path, "all_files")

    def rm_set_dir(self):
        for set_dir in ["test", "train", "valid"]:
            set_dir_path = os.path.join(self.db_to_clean_path, set_dir)
            if os.path.isdir(set_dir_path):
                shutil.rmtree(set_dir_path)

    def manage_unique_set(self, set_dir):
        if set_dir == "all_files":
            return
        complete_set_dir = os.path.join(self.db_to_clean_path, set_dir)
        if not os.path.isdir(complete_set_dir):  # Si le dir n'existe pas, on passe au suivant
            return
        for bird_dir in tqdm(os.listdir(complete_set_dir), desc="Parcours set " + set_dir):
            # Parcourt de chaque oiseau
            new_bird_dir = " ".join(os.path.join(self.all_file_path, bird_dir).split())
            # On retire les espaces en trop

            if not os.path.isdir(new_bird_dir):
                # Creation du nouveau dossier oiseau dans lesquel les autres seront fusionnés
                os.mkdir(new_bird_dir)
            complete_bird_dir = os.path.join(complete_set_dir, bird_dir)
            for file in os.listdir(complete_bird_dir):
                # Parcourt de chaque fichier
                if not os.path.isfile(os.path.join(complete_bird_dir, file)):
                    continue
                part_name = int(file.split(".")[0])
                # On récupère le nom du fichier qui est un nombre pour ne pas avoir de doublon
                while os.path.isfile(os.path.join(new_bird_dir, str(part_name) + ".jpg")):
                    # Le temps qu'un fichier de même nombre existe, on incrémente le nombre
                    part_name += 1
                os.rename(
                    os.path.join(complete_bird_dir, file),  # On déplace le fichier
                    os.path.join(new_bird_dir, str(part_name) + ".jpg"),
                )

    def sets_fusion(self):
        print("Fusion des sets test, train, valid")
        if not os.path.isdir(self.all_file_path):
            # S'il n'existe pas déjà, création d'un dossier dans lequel fusionner les sets par oiseau
            os.mkdir(self.all_file_path)
        elif (
            not os.path.isdir(os.path.join(self.db_to_clean_path, "train")) and not
            os.path.isdir(os.path.join(self.db_to_clean_path, "test")) and not
            os.path.isdir(os.path.join(self.db_to_clean_path, "valid"))
        ):
            print("Les dossiers sont déjà fusionnés")
            return

        for set_dir in os.listdir(self.db_to_clean_path):  # Parcourt de chaque set
            self.manage_unique_set(set_dir)
        # Une fois la fusion terminée, on supprimer les dossiers test, train et valid
        self.rm_set_dir()

    def split_train_test_valid(self, percent=15):
        print("Debut du split pour les set test et valid")
        # Calcul de 15% de la première classe. Comme samplé, égal pour toutes les classes
        percent_number_of_files = self.calcul_percent_number(percent)

        self.split_set_class_balancing("test", percent_number_of_files)
        self.split_set_class_balancing("valid", percent_number_of_files)
        os.rename(self.all_file_path, os.path.join(self.db_to_clean_path, "train"))

    def move_one_random_file(self, nb_classes, all_classes, set_path):
        # On choisit une classe au hasard, et une image de la classe au hasard
        rand_dir = random.randint(0, nb_classes - 1)  # Choix d'une classe au hasard
        dir_class_name = all_classes[rand_dir]  # Nom de la classe
        choosen_class_path = os.path.join(
            self.all_file_path, dir_class_name
        )  # chemin de la classe dans les set fusionnés
        files_class = os.listdir(choosen_class_path)  # Récupération de tout les fichiers de la classe
        rand_file = random.randint(0, len(files_class) - 1)  # Choix d'un fichier au hasard
        file_name = files_class[rand_file]  # Nom du fichier
        choosen_file_path = os.path.join(choosen_class_path, file_name)  # Chemin du fichier
        new_class_path = os.path.join(set_path, dir_class_name)  # Chemin de la class dans le nouveau set
        if not os.path.isdir(new_class_path):  # Creation du nouveau dossier oiseau dans le set s'il n'existe pas
            os.mkdir(new_class_path)
        os.rename(
            choosen_file_path, os.path.join(new_class_path, file_name)
        )  # Déplacement du fichier dans le nouveau set

    def split_set_random_pull(self, set_name, percent=15):
        # Version 15% du total des images
        if self.random_state:
            random.seed(12)
        if set_name not in ["valid", "test"]:
            print("Erreur de nom de set")
            return

        set_path = os.path.join(self.db_to_clean_path, set_name)
        if not os.path.isdir(set_path):
            # Creation du nouveau dossier set
            os.mkdir(set_path)

        nb_files = sum([len(files) for r, c, files in os.walk(self.all_file_path)])  # nombre de fichier du dataset
        all_classes = os.listdir(self.all_file_path)  # Liste des classes
        nb_classes = len(all_classes)  # Nombre de classe
        if percent < 100:
            percent = int((nb_files / 100) * percent)  # Calcul du nombre de fichier pour avoir 15% du dataset (arrondi)

        for i in tqdm(range(percent), "Split du set " + set_name):  # 15%
            self.move_one_random_file(nb_classes, all_classes, all_classes, set_path)
        return percent

    def calcul_percent_number(self, percent):
        all_classes = os.listdir(self.all_file_path)
        one_class_path = os.path.join(self.all_file_path, all_classes[0])  # Le chemin de la première classe
        files_class = os.listdir(one_class_path)  # Récupération de tout les fichiers de la classe
        percent_number_of_files = int(
            (len(files_class) / 100) * percent
        )  # Calcul du nombre de fichier pour avoir 15% du dataset (arrondi)
        return percent_number_of_files

    def extract_percent_from_one_class(self, classe_index, all_classes, set_path, percent):
        dir_class_name = all_classes[classe_index]  # Nom de la classe
        dir_class_path = os.path.join(self.all_file_path, dir_class_name)  # chemin de la classe dans les set fusionnés
        files_class = os.listdir(dir_class_path)  # Récupération de tout les fichiers de la classe
        for i in range(percent):
            rand_file = random.randint(0, len(files_class) - 1)  # Choix d'un fichier au hasard
            file_name = files_class[rand_file]  # Nom du fichier
            choosen_file_path = os.path.join(dir_class_path, file_name)  # Chemin du fichier
            new_class_path = os.path.join(set_path, dir_class_name)  # Chemin de la class dans le nouveau set
            if not os.path.isdir(new_class_path):  # Creation du nouveau dossier oiseau dans le set s'il n'existe pas
                os.mkdir(new_class_path)
            os.rename(
                choosen_file_path, os.path.join(new_class_path, file_name)
            )  # Déplacement du fichier dans le nouveau set
            del files_class[rand_file]

    def split_set_class_balancing(self, set_name, percent):
        # Version 15% de chaque classe
        if self.random_state:
            random.seed(12)
        if set_name not in ["valid", "test"]:
            print("Erreur de nom de set")
            return
        set_path = os.path.join(self.db_to_clean_path, set_name)
        if not os.path.isdir(set_path):
            # Creation du nouveau dossier set
            os.mkdir(set_path)

        all_classes = os.listdir(self.all_file_path)  # Liste des classes
        nb_classes = len(all_classes)  # Nombre de classe

        for classe_index in tqdm(range(nb_classes), "Split du set " + set_name):
            self.extract_percent_from_one_class(classe_index, all_classes, set_path, percent)

    def start_clean(self):
        self.dataset_correction()
        sizeManager = SizeManager(db_to_clean_path=self.db_to_clean_path)
        sizeManager.manage()
        self.sets_fusion()
        self.under_sample()
        self.split_train_test_valid()
        self.check_percents()

    def dataset_correction(self):
        """
        Correction des incoherences des données, et création du fichier CSV de modèle d'espèces
        """
        datasetCorrection = DatasetCorrection(db_to_clean=self.db_to_clean_path, test_mode=self.test_mode)
        datasetCorrection.full_correction()

    def under_sample(self):
        underSampler = UnderSamplerImages(self.db_to_clean_path, treshold=self.treshold)
        underSampler.check_distribution()
        underSampler.del_under_treshold_classes()
        underSampler.under_sample()
        underSampler.check_distribution()

    def cleanAll(self):
        if not os.path.isdir(self.db_to_clean_path):
            print(
                "L'exécution correcte de ce script requiert la présence du dossier : ",
                self.db_to_clean_path,
            )
            print("Dossier non trouvé")
            return
        start_time = time.time()
        self.start_clean()
        end_time = time.time()
        print(f"Le temps d'exécution est {(end_time - start_time)/60} minutes.")

    def check_percents(self):
        prop_set = dict()
        for set_name in os.listdir(self.db_to_clean_path):
            set_path = os.path.join(self.db_to_clean_path, set_name)
            prop_set[set_name] = 0
            if os.path.isdir(set_path):
                for dir in os.listdir(set_path):
                    dir_class = os.path.join(set_path, dir)
                    prop_set[set_name] += len(os.listdir(dir_class))
        for set_name in prop_set:
            if os.path.isdir(os.path.join(self.db_to_clean_path, set_name)):
                total_db = sum(prop_set.values())
                total_set = int(prop_set[set_name])
                print(
                    "Proportion d'image dans ",
                    set_name,
                    " : %.2f" % (total_set / total_db),
                )
                print("Total set ", set_name, " : ", total_set)


if __name__ == "__main__":

    cleanDB = CleanDB("./data", treshold=False, test_mode=False)
    # cleanDB = CleanDB(const_prod.DATASET_TEST, treshold=False)
    # Le treshold indique combien d'images il y aura pour chaque classe.
    # La valeur par défaut est 160 pour donner 394 classes
    cleanDB.cleanAll()
