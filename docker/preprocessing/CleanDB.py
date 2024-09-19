import os
import random
import shutil
import time
from tqdm import tqdm
from UnderSampling import UnderSamplerImages
from SizeManager import SizeManager


class CleanDB:
    def __init__(self, db_to_clean, treshold=160, random_state=True, test_mode: bool = False):
        """
        On initialise les chemins, le seuil, la variable aléatoire ainsi que la possibilité d'activer le mode test.
        """
        # Chemin vers la base de données à nettoyer
        self.db_to_clean_path = db_to_clean
        # Permet une certaine reproductibilité
        self.random_state = random_state
        # Seuil pour sous échantillonnage
        self.treshold = treshold
        # Mode test (optionnel)
        self.test_mode = test_mode
        # Chemin vers les dossiers fusionnés pour la re répartition des classes
        self.all_file_path = os.path.join(self.db_to_clean_path, "all_files")

    def rm_set_dir(self):
        """
        Suppression des dossiers test, train et valid une fois la fusion terminée.
        On parcourt chaque set, et si le dossier existe, on le supprime.
        """
        # Liste des dossiers à supprimer
        for set_dir in ["test", "train", "valid"]:
            set_dir_path = os.path.join(self.db_to_clean_path, set_dir)
            # Si le dossier existe, on le supprime
            if os.path.isdir(set_dir_path):
                shutil.rmtree(set_dir_path)

    def manage_unique_set(self, set_dir):
        """
        Fusionne les sets test, train et valid vers le dossier 'all_files'.
        On renomme les fichiers s'il y a des conflits.
        """
        if set_dir == "all_files":
            return
        complete_set_dir = os.path.join(self.db_to_clean_path, set_dir)
        # Si le dossier n'existe pas, on stop
        if not os.path.isdir(complete_set_dir):
            return
        # On parcours chaque classe
        for bird_dir in tqdm(os.listdir(complete_set_dir), desc="Parcours set " + set_dir):
            # On retire les espaces en trop dans le chemin du dossier
            new_bird_dir = " ".join(os.path.join(self.all_file_path, bird_dir).split())

            # On créer un dossier pour la classe dans all_files s'il n'existe pas
            if not os.path.isdir(new_bird_dir):
                os.mkdir(new_bird_dir)
            complete_bird_dir = os.path.join(complete_set_dir, bird_dir)
            # Parcours de chaque image dans la classe
            for file in os.listdir(complete_bird_dir):
                # Si ce n'est pas une image, on continue
                if not os.path.isfile(os.path.join(complete_bird_dir, file)):
                    continue
                # On récupère le nombre du fichier
                part_name = file.split(".")[0]
                # On vérifie si le nom est un nombre (si ce n'est pas le cas,
                # c'est un hash, donc pas de doublons à gérer)
                if part_name.isdigit():
                    part_name = int(part_name)
                    # On gère les doublons en incrémentant le nombre
                    while os.path.isfile(os.path.join(new_bird_dir, str(part_name) + ".jpg")):
                        part_name += 1
                # On déplace le fichier dans le dossier complet
                os.rename(
                    os.path.join(complete_bird_dir, file),
                    os.path.join(new_bird_dir, str(part_name) + ".jpg"),
                )

    def sets_fusion(self):
        """
        Fusionne test, train et valid dans le dossier all_files.
        """
        print("Fusion des sets test, train, valid")
        # On créer le dossier all_files s'il nexiste pas
        if not os.path.isdir(self.all_file_path):
            os.mkdir(self.all_file_path)
        # Si les dossiers sont déjà fusionnés, on arrête
        elif (
            not os.path.isdir(os.path.join(self.db_to_clean_path, "train"))
            and not os.path.isdir(os.path.join(self.db_to_clean_path, "test"))
            and not os.path.isdir(os.path.join(self.db_to_clean_path, "valid"))
        ):
            print("Les dossiers sont déjà fusionnés")
            return

        # Fusion de chaque set
        for set_dir in os.listdir(self.db_to_clean_path):
            self.manage_unique_set(set_dir)
        # Suppression des dossiers test, train et valid une fois la fusion faite
        self.rm_set_dir()

    def split_train_test_valid(self, percent=15):
        """
        Split des données entre test et valid, avec un pourcentage donné.
        """
        print("Debut du split pour les set test et valid")
        # Calcul du nombre de fichiers à déplacer
        percent_number_of_files = self.calcul_percent_number(percent)

        # On réalise la répartition
        self.split_set_class_balancing("test", percent_number_of_files)
        self.split_set_class_balancing("valid", percent_number_of_files)
        # On renomme le dossier all_files en train
        os.rename(self.all_file_path, os.path.join(self.db_to_clean_path, "train"))

    def move_one_random_file(self, nb_classes, all_classes, set_path):
        """
        Déplace un fichier aléatoire d'une classe choisie au hasard vers un set donné.
        """
        # On sélectionne une classe au hasard
        rand_dir = random.randint(0, nb_classes - 1)
        # On récupère son nom
        dir_class_name = all_classes[rand_dir]
        choosen_class_path = os.path.join(self.all_file_path, dir_class_name)

        # On liste les images de la classe
        files_class = os.listdir(choosen_class_path)
        # On choisis une image au hasard
        rand_file = random.randint(0, len(files_class) - 1)
        file_name = files_class[rand_file]
        choosen_file_path = os.path.join(choosen_class_path, file_name)
        new_class_path = os.path.join(set_path, dir_class_name)

        # On créer le dossier cible s'il n'existe pas
        if not os.path.isdir(new_class_path):
            os.mkdir(new_class_path)
        # On déplace l'image
        os.rename(choosen_file_path, os.path.join(new_class_path, file_name))

    def split_set_random_pull(self, set_name, percent=15):
        """
        Répartition aléatoire d'un pourcentage total d'images (par défaut 15%) dans un set.
        Ce split n'est pas équilibré entre les classes.
        """
        # On choisi un état aléatoire
        if self.random_state:
            random.seed(12)
        # Si le nom n'est pas valide, on stop
        if set_name not in ["valid", "test"]:
            print("Erreur de nom de set")
            return

        # Création du dossier du set s'il n'existe pas
        set_path = os.path.join(self.db_to_clean_path, set_name)
        if not os.path.isdir(set_path):
            os.mkdir(set_path)

        # On récupère le nombre total de fichiers
        nb_files = sum([len(files) for r, c, files in os.walk(self.all_file_path)])
        # On liste les classes
        all_classes = os.listdir(self.all_file_path)
        # On récupère le nombre de classes
        nb_classes = len(all_classes)
        # On calcul le nombre d'images à déplacer
        if percent < 100:
            percent = int((nb_files / 100) * percent)

        # On déplace aléatoiement des images jusuq'à atteindre le pourcentage voulu
        for i in tqdm(range(percent), "Split du set " + set_name):
            self.move_one_random_file(nb_classes, all_classes, all_classes, set_path)
        return percent

    def calcul_percent_number(self, percent):
        """
        Calcule le nombre de fichiers à déplacer pour un pourcentage donné, basé sur la taille d'une seule classe.
        """
        all_classes = os.listdir(self.all_file_path)
        # Chemin vers la première classe
        one_class_path = os.path.join(self.all_file_path, all_classes[0])
        # On liste les images de la première classe de all_class
        files_class = os.listdir(one_class_path)
        # On calcul le nombre de fichiers en fonction du pourcentage
        percent_number_of_files = int((len(files_class) / 100) * percent)
        return percent_number_of_files

    def extract_percent_from_one_class(self, classe_index, all_classes, set_path, percent):
        """
        Déplace un pourcentage d'images d'une seule classe vers un set donné. Les images sont choisies au hasard.
        """
        # On récupère le nom de la classe et son chemin
        dir_class_name = all_classes[classe_index]
        dir_class_path = os.path.join(self.all_file_path, dir_class_name)
        # On liste les fichiers de la classe
        files_class = os.listdir(dir_class_path)
        # On déplace les images en fonction du pourcentage défini
        for i in range(percent):
            # On sélectionne une image au hasard
            rand_file = random.randint(0, len(files_class) - 1)
            file_name = files_class[rand_file]
            choosen_file_path = os.path.join(dir_class_path, file_name)
            new_class_path = os.path.join(set_path, dir_class_name)

            # On créer le dossier de destination s'il n'existe pas
            if not os.path.isdir(new_class_path):
                os.mkdir(new_class_path)
            # On déplace le l'image
            os.rename(choosen_file_path, os.path.join(new_class_path, file_name))
            # On supprime l'image de la liste
            del files_class[rand_file]

    def split_set_class_balancing(self, set_name, percent):
        """
        Répartition équilibrée de chaque classe dans un set.
        On déplace un pourcentage d'images pour chaque classe.
        """
        # On fixe l'état aléatoire pour la reproductibilité
        if self.random_state:
            random.seed(12)
        # Si le set n'est pas valide, on s'arrête
        if set_name not in ["valid", "test"]:
            print("Erreur de nom de set")
            return
        set_path = os.path.join(self.db_to_clean_path, set_name)
        # On créer le set s'il n'existe pas
        if not os.path.isdir(set_path):
            os.mkdir(set_path)

        # On liste toutes les classes
        all_classes = os.listdir(self.all_file_path)
        # On récupère le nombre total d'images
        nb_classes = len(all_classes)

        # On fait la répartition des images
        for classe_index in tqdm(range(nb_classes), "Split du set " + set_name):
            self.extract_percent_from_one_class(classe_index, all_classes, set_path, percent)

    def start_clean(self):
        """
        Démarre la procédure complète de nettoyage du dataset.
        """
        # On instancie la classe
        sizeManager = SizeManager(db_to_clean_path=self.db_to_clean_path)
        # On ajuste la taille des images si nécessaire
        sizeManager.manage()
        # On fusionne les sets
        self.sets_fusion()
        # On sous échantillonne les classes si nécessaire
        self.under_sample()
        # On crée les sets de train, test et valid
        self.split_train_test_valid()
        # On vérifie les pourcentages pour la répartition
        self.check_percents()

    def under_sample(self):
        """
        Applique un sous-échantillonnage pour retirer les classes avec un nombre d'images inférieur au seuil défini.
        """
        # On instancie la classe
        underSampler = UnderSamplerImages(self.db_to_clean_path, treshold=self.treshold)
        # On vérifie la distribution des classes
        underSampler.check_distribution()
        # On supprime les classes sous représentées
        underSampler.del_under_treshold_classes()
        # On applique le sous échantillonnage
        underSampler.under_sample()
        # On vérifie la distribution
        underSampler.check_distribution()

    def cleanAll(self):
        """
        Nettoie toute la base de données.
        """
        if not os.path.isdir(self.db_to_clean_path):
            print(
                "L'exécution correcte de ce script requiert la présence du dossier : ",
                self.db_to_clean_path,
            )
            print("Dossier non trouvé")
            return
        # On calcul le temps d'exécution
        start_time = time.time()
        self.start_clean()
        end_time = time.time()
        print(f"Le temps d'exécution est {(end_time - start_time)/60} minutes.")  # Affiche le temps d'exécution

    def check_percents(self):
        """
        Vérifie les proportions d'images dans train, test et valid.
        Affiche les résultats pour chaque set et la taille totale de celui-ci.
        """
        # On crée un dictionnaire pour stocker les proportions
        prop_set = dict()
        # On parcours les sets
        for set_name in os.listdir(self.db_to_clean_path):
            set_path = os.path.join(self.db_to_clean_path, set_name)
            # On initialise le compteur pour chaque set
            prop_set[set_name] = 0
            if os.path.isdir(set_path):
                # On compte le nombre d'images dans chaque classe du set
                for dir in os.listdir(set_path):
                    dir_class = os.path.join(set_path, dir)
                    prop_set[set_name] += len(os.listdir(dir_class))
        # On affiche les proportions pour chaque set
        for set_name in prop_set:
            if os.path.isdir(os.path.join(self.db_to_clean_path, set_name)):
                # Total d'images dans la base de données
                total_db = sum(prop_set.values())
                # Total d'images dans le set actuel
                total_set = int(prop_set[set_name])
                print(
                    "Proportion d'image dans ",
                    set_name,
                    " : %.2f" % (total_set / total_db),
                )
                print("Total set ", set_name, " : ", total_set)
