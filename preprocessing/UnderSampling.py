import os
import sys
import random
import shutil


class UnderSamplerImages:
    def __init__(self, root_dir, treshold=False):
        self.root_dir = root_dir

        if treshold is False:
            self.treshold = self.get_min_size()  # nombre maximum d'image de train pour une classe
        else:
            self.treshold = treshold

    def get_min_size(
        self,
    ):  # On cherche la classe aec le plus d'image pour aligner les autres dessus
        classes = os.listdir(os.path.join(self.root_dir, "all_files"))
        min_size_class = sys.maxsize
        for classe in classes:
            size_class = len(os.listdir(os.path.join(self.root_dir, "all_files", classe)))
            if size_class < min_size_class:
                min_size_class = size_class
        return min_size_class

    def under_sample(self):
        all_files_path = os.path.join(self.root_dir, "all_files")
        for classe in os.listdir(all_files_path):
            classe_path = os.path.join(all_files_path, classe)
            files = os.listdir(classe_path)
            while len(files) > self.treshold:
                # tant qu'il y a plus d'image que le treshold, on supprime des images
                index_to_del = random.randint(0, len(files) - 1)

                os.remove(os.path.join(classe_path, files[index_to_del]))
                del files[index_to_del]

    def del_under_treshold_classes(self):
        # Supprime les classes du dossier all_files qui ont un nombre de files < treshold
        all_files_path = os.path.join(self.root_dir, "all_files")
        count = 0
        for classe in os.listdir(all_files_path):
            classe_path = os.path.join(all_files_path, classe)
            files = os.listdir(classe_path)
            if len(files) < self.treshold:
                # si il y a moins d'images que le seuil on supprimer la classe
                if os.path.exists(classe_path):
                    shutil.rmtree(classe_path)
                    count += 1
        print("Classes deleted : ", count)

    def check_distribution(self):
        # Affiche le nombre d'image par classe
        print("Valeur seuille : ", self.treshold)
        all_files_path = os.path.join(self.root_dir, "all_files")
        if not os.path.isdir(all_files_path):
            return
        classes = os.listdir(all_files_path)
        nb_image_by_classe = list()
        for classe in classes:
            if os.path.isdir(os.path.join(all_files_path, classe)):
                size_classe = len(os.listdir(os.path.join(all_files_path, classe)))
                if size_classe > self.treshold:
                    nb_image_by_classe.append(size_classe - self.treshold)
        print("Il y a ", len(nb_image_by_classe), " classe(s) à diminuer")
        print("Il faut supprimer ", sum(nb_image_by_classe), " image(s)")


if __name__ == "__main__":
    overSampler = UnderSamplerImages("./data", treshold=170)
    overSampler.check_distribution()
    # overSampler.under_sample()
    overSampler.check_distribution()
    overSampler.del_under_treshold_classes()
