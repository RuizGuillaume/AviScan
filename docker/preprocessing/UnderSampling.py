import os
import sys
import random
import shutil


class UnderSamplerImages:
    """
    Classe permettant d'effectuer le sous-échantillonage du dataset.
    On s'assure que chaque classe possède un nombre maximum d'images,
    en supprimant les images en trop. Si à l'inverse une classe a trop
    peu d'images, on la supprime.
    """
    def __init__(self, root_dir, treshold=False):
        self.root_dir = root_dir
        # Si aucun seuil n'est défini, on récupère le nombre d'images de la classe la plus petite
        if treshold is False:
            self.treshold = self.get_min_size()
        else:
            # Si un seuil est donné, on l'utilise
            self.treshold = treshold

    def get_min_size(self):
        """
        On cherche la classe qui a le moins d'images
        """
        classes = os.listdir(os.path.join(self.root_dir, "all_files"))
        # On initialise la variable avec le plus grand nombre possible
        min_size_class = sys.maxsize
        for classe in classes:
            # On récupère le nombre d'images de chaque classe
            size_class = len(os.listdir(os.path.join(self.root_dir, "all_files", classe)))
            # Si une classe a un nombre d'images inférieur à la précédente,
            # elle devient la plus petite classe
            if size_class < min_size_class:
                min_size_class = size_class
        return min_size_class

    def under_sample(self):
        """
        On réduit le nombre d'images par classe si elles dépassent le seuil défini
        """
        all_files_path = os.path.join(self.root_dir, "all_files")
        # On parcours chaque classe
        for classe in os.listdir(all_files_path):
            classe_path = os.path.join(all_files_path, classe)
            files = os.listdir(classe_path)
            # Tant qu'une classe a plus d'images que le seuil, on en supprime
            while len(files) > self.treshold:
                # On choisit une image au hasard
                index_to_del = random.randint(0, len(files) - 1)
                # On supprime l'image
                os.remove(os.path.join(classe_path, files[index_to_del]))
                # On met à jour la liste
                del files[index_to_del]

    def del_under_treshold_classes(self):
        """
        Supprime les classes qui ont moins d'images que le seuil défini
        """

        all_files_path = os.path.join(self.root_dir, "all_files")
        # Compteur pour suivre combien de classes sont supprimées
        count = 0
        # On parcours les classes
        for classe in os.listdir(all_files_path):
            classe_path = os.path.join(all_files_path, classe)
            files = os.listdir(classe_path)
            # Si une classe a moins d'images que le seuil, on la supprime
            if len(files) < self.treshold:
                if os.path.exists(classe_path):
                    # Suppression de la classe
                    shutil.rmtree(classe_path)
                    # On incrémente le compteur
                    count += 1
        print("Classes deleted : ", count)

    def check_distribution(self):
        """
        Affiche le nombre d'images par classe et vérifie combien dépassent le seuil
        """
        print("Valeur seuille : ", self.treshold)
        all_files_path = os.path.join(self.root_dir, "all_files")
        # Si le dossier n'existe pas, on s'arrête
        if not os.path.isdir(all_files_path):
            return
        classes = os.listdir(all_files_path)
        # On stocke le nombre d'images par classe
        nb_image_by_classe = list()
        # On parcours chaque classe
        for classe in classes:
            if os.path.isdir(os.path.join(all_files_path, classe)):
                # On récupère le nombre d'images de la classe
                size_classe = len(os.listdir(os.path.join(all_files_path, classe)))
                # Si une classe dépasse le seuil, on ajoute l'excédent dans la liste
                if size_classe > self.treshold:
                    nb_image_by_classe.append(size_classe - self.treshold)
        print("Il y a ", len(nb_image_by_classe), " classe(s) à diminuer")
        print("Il faut supprimer ", sum(nb_image_by_classe), " image(s)")


# Exemple si utilisé directement
if __name__ == "__main__":
    # On instancie la classe avec un dossier et un seuil
    overSampler = UnderSamplerImages("./data", treshold=170)
    # On vérifie la distribution avant de faire des modifications
    overSampler.check_distribution()
    # overSampler.under_sample()  # Permet de faire l'undersampling
    # On vérifie à nouveau la distribution
    overSampler.check_distribution()
    # On supprime les classes avec trop peu d'images
    overSampler.del_under_treshold_classes()
