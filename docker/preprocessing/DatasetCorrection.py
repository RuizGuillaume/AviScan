import pandas as pd
import requests
import os
import shutil


class DatasetCorrection:

    def __init__(self, db_to_clean, test_mode: bool = False):
        """
        Initialisation de l'objet
        """
        self.db_to_clean = db_to_clean
        self.test_mode = test_mode

    def download_full_dataset(self):
        """
        Télécharge un fichier csv contenant les noms vernaculaires sans fautes et noms binominaux à jours
        """
        link = "https://worldbirdnames.org/Multiling%20IOC%2014.1_b.xlsx"
        response = requests.get(
            link
        )  # Télechargement du fichier excel, répertoriant beaucoup d'espèces (11194 espèces au 31/07/24)
        with open(f"{self.db_to_clean}/birds_list.xlsx", "wb") as f:
            f.write(response.content)

        birds_list_df = pd.read_excel(f"{self.db_to_clean}/birds_list.xlsx")

        birds_list_df = birds_list_df[
            ["IOC_14.1", "English", "French"]
        ]  # On sélectionne les colonnes (dont les langues) à conserver
        birds_list_df = birds_list_df.rename(columns={"IOC_14.1": "BinomialNomenclature"})

        for (
            column
        ) in (
            birds_list_df.columns
        ):  # Pour chaque langue, on enlève les espaces parasites et on passe en minuscules sauf la premiere lettre.
            if column == "BinomialNomenclature":
                continue
            birds_list_df[column] = birds_list_df[column].astype(str).apply(lambda row: self.correctString(row))

        birds_list_df = birds_list_df.drop_duplicates()
        birds_list_df = birds_list_df.dropna()
        birds_list_df = birds_list_df.reset_index()

        birds_list_df.to_csv(
            f"{self.db_to_clean}/birds_list.csv", index=False
        )  # Fichier csv répertoriant la liste d'espèces à proposer aux utilisateurs
        os.remove(f"{self.db_to_clean}/birds_list.xlsx")

    def test_phase_init(self):
        """
        Génère un dataset de x images maximum par classe et met à jour le fichier birds.csv
        """
        images_per_class_to_keep = 5

        if os.path.exists(f"{self.db_to_clean}/birds_backup.csv"):
            os.remove(f"{self.db_to_clean}/birds_backup.csv")
        os.rename(
            f"{self.db_to_clean}/birds.csv", f"{self.db_to_clean}/birds_backup.csv"
        )  # Pour le test, on renomme le csv pour générer un backup
        with open(f"{self.db_to_clean}/birds_backup.csv", "r") as f_in, open(
            f"{self.db_to_clean}/birds.csv", "w"
        ) as f_out:
            first_line = True
            for row in f_in:
                if first_line:  # La première ligne correspond au header et est  recopié
                    first_line = False
                    f_out.write(row)
                    continue

                # On extrait le nom du fichier (000 à 999, sans extension)
                _, path, _, _, _ = row.split(",")
                _, _, filename = path.split("/")
                filename, _ = filename.split(".")

                # On ne garde que les premières images pour réduire la taille du dataset pour les tests
                if int(filename) <= images_per_class_to_keep:
                    f_out.write(row)  # On conserve le fichier dans le nouveau csv
                else:
                    try:
                        os.remove(f"{self.db_to_clean}/{path}")  # Suppression du fichier
                    except Exception:
                        print(f"Fichier manquant : {path}")

    def dataset_correction(self):
        """
        Correction du dataset (harmonisation des noms binominaux et vernaculaires anglais, correction de fautes,
        gestion des races, correction des chemins, création du nouveau csv)
        """

        df = pd.read_csv(f"{self.db_to_clean}/birds.csv", sep=",")
        birds_list_df = pd.read_csv(f"{self.db_to_clean}/birds_list.csv", index_col="index", sep=",")

        # region Corrections manuelles du fichier Bird.csv
        df.loc[df["labels"] == "BANDED PITA", "labels"] = "PITTA"
        df.loc[df["labels"] == "COCK OF THE  ROCK", "labels"] = "COCK OF THE ROCK"
        df.loc[df["labels"] == "TOUCHAN", "labels"] = "TOUCAN"
        df.loc[df["labels"] == "AMERICAN AVOCET", "scientific name"] = "RECURVIROSTRA AMERICANA"
        df.loc[df["labels"] == "GILDED FLICKER", "scientific name"] = "COLAPTES CHRYSOIDES"
        df.loc[df["scientific name"] == "TOCKUS FASCIATUS", "scientific name"] = "LOPHOCEROS FASCIATUS"
        df.loc[df["scientific name"] == "EUPHONIA MUSICA", "scientific name"] = "CHLOROPHONIA MUSICA"
        df.loc[df["scientific name"] == "HYDRORNIS GUAJANA", "scientific name"] = "HYDRORNIS"
        df.loc[df["scientific name"] == "CICINNURUS RESPUBLICA", "scientific name"] = "DIPHYLLODES RESPUBLICA"
        df.loc[df["scientific name"] == "STRELITZIA", "scientific name"] = "PARADISAEA"
        df.loc[df["scientific name"] == "CALYPTORHYNCHUS LATIROSTRIS", "scientific name"] = "ZANDA LATIROSTRIS"
        df.loc[df["scientific name"] == "AMAURORNIS BICOLOR", "scientific name"] = "ZAPORNIA BICOLOR"
        df.loc[df["scientific name"] == "PUFFINUS OPISTHOMELA", "scientific name"] = "PUFFINUS OPISTHOMELAS"
        df.loc[df["scientific name"] == "PHALACROCORAX PENICILLATUS", "scientific name"] = "URILE PENICILLATUS"
        df.loc[df["scientific name"] == "SERINUS CANARIA DOMESTICA", "scientific name"] = "SERINUS CANARIA"
        df.loc[df["scientific name"] == "CACATUIDAE", "scientific name"] = "CACATUA"
        df.loc[df["scientific name"] == "PTEROGLOSSUS BEAUHARNAESII", "scientific name"] = "PTEROGLOSSUS BEAUHARNAISII"
        df.loc[df["scientific name"] == "TAENIOPYGIA BICHENOVII", "scientific name"] = "STIZOPTERA BICHENOVII"
        df.loc[df["scientific name"] == "PHALACROCORAX AURITUS", "scientific name"] = "NANNOPTERUM AURITUM"
        df.loc[df["scientific name"] == "IRENA", "scientific name"] = "IRENA PUELLA"
        df.loc[df["scientific name"] == "FREGATIDAE", "scientific name"] = "FREGATA"
        df.loc[df["scientific name"] == "COLUMBA LIVIA DOMESTICA", "scientific name"] = "COLUMBA LIVIA"
        df.loc[df["scientific name"] == "CORYTHAIXOIDES CONCOLOR", "scientific name"] = "CRINIFER CONCOLOR"
        df.loc[df["scientific name"] == "ICHTHYOPHAGA ICHTHYAETUS", "scientific name"] = "ICTHYOPHAGA ICHTHYAETUS"
        df.loc[df["scientific name"] == "UPUPIDAE", "scientific name"] = "UPUPA"
        df.loc[df["scientific name"] == " VESTIARIA COCCINEA.", "scientific name"] = "DREPANIS COCCINEA"
        df.loc[df["scientific name"] == " FRATERCULA", "scientific name"] = "FRATERCULA ARCTICA"
        df.loc[df["scientific name"] == "PITTA ERYTHROGASTER", "scientific name"] = "ERYTHROGASTER"
        df.loc[df["scientific name"] == "PHAETHON AETHEREU", "scientific name"] = "PHAETHON AETHEREUS"
        df.loc[df["scientific name"] == "PHALACROCORAX URILE", "scientific name"] = "URILE URILE"
        df.loc[df["scientific name"] == "REGULUS CALENDULA", "scientific name"] = "CORTHYLIO CALENDULA"
        df.loc[df["scientific name"] == "AMAURORNIS CINEREA", "scientific name"] = "POLIOLIMNAS CINEREUS"
        df.loc[df["scientific name"] == "TAURACO LEUCOTIS", "scientific name"] = "MENELIKORNIS LEUCOTIS"
        df.loc[df["scientific name"] == "TROPICRANUS ALBOCRISTATUS", "scientific name"] = "HORIZOCERUS ALBOCRISTATUS"
        df.loc[df["scientific name"] == "CICINNURUS RESPUBLICA", "scientific name"] = "DIPHYLLODES RESPUBLICA"
        df.loc[df["scientific name"] == "DICAEUM MELANOXANTHUM", "scientific name"] = "DICAEUM MELANOZANTHUM"
        df.loc[df["scientific name"] == "XANTHOCEPHALUS", "scientific name"] = "XANTHOCEPHALUS XANTHOCEPHALUS"
        df["filepaths"] = df["filepaths"].str.replace("train/PARAKETT  AKULET", "train/PARAKETT  AUKLET")
        df["filepaths"] = df["filepaths"].str.replace("test/PARAKETT  AKULET", "test/PARAKETT  AUKLET")
        df["filepaths"] = df["filepaths"].str.replace("valid/PARAKETT  AKULET", "valid/PARAKETT AUKLET")
        # endregion

        # region Espèces à supprimer
        species_to_delete = ["LOONEY BIRDS"]
        for species in species_to_delete:
            df = df[df["labels"] != species]
            for folder in ["train", "test", "valid"]:
                path = f"{self.db_to_clean}/{folder}/{species}"  # exemple : ./data/raw/test/LOONEY BIRDS
                if os.path.isdir(path):
                    shutil.rmtree(path)  # Si le dossier existe, on le supprime
        # endregion

        # region Liste des races à corriger à posteriori du traitement
        dict_breed_manual_correction = {
            "Jacobin pigeon": df.loc[df["labels"] == "JACOBIN PIGEON", "class id"].iloc[0],
            "Frillback pigeon": df.loc[df["labels"] == "FRILL BACK PIGEON", "class id"].iloc[0],
        }
        # endregion

        # region Harmonisation des noms vernaculaires et/ou binomiaux des noms des dossiers et dans le fichier birds.csv
        # On applique capitalize à nos colonnes de noms (vernaculaires et binominaux), et on retire les apostrophes
        birds_list_df["English"] = birds_list_df["English"].astype(str).apply(lambda row: self.correctString(row))
        birds_list_df["BinomialNomenclature"] = (
            birds_list_df["BinomialNomenclature"].astype(str).apply(lambda row: self.correctString(row))
        )
        df["labels"] = df["labels"].astype(str).apply(lambda row: self.correctString(row))
        df["scientific name"] = df["scientific name"].astype(str).apply(lambda row: self.correctString(row))

        # Création d'un dictionnaire pour la correspondance
        binomial_to_english = dict(zip(birds_list_df["BinomialNomenclature"], birds_list_df["English"]))
        english_to_binomial = dict(zip(birds_list_df["English"], birds_list_df["BinomialNomenclature"]))

        # Mise à jour des colonnes 'labels' et 'scientific name', priorité au fichier birds_list
        df["labels"] = df.apply(
            lambda row: binomial_to_english.get(row["scientific name"], row["labels"]),
            axis=1,
        )
        df["scientific name"] = df.apply(
            lambda row: english_to_binomial.get(row["labels"], row["scientific name"]),
            axis=1,
        )

        # Correction manuel des races
        for breed_key, breed_value in dict_breed_manual_correction.items():
            df.loc[df["class id"] == breed_value, "labels"] = breed_key

        def adapt_file_path(row):
            """
            Retourne le contenu de la colonne filepaths avec le nouveau label
            """
            path = row["filepaths"]
            true_label = row["labels"]  # On récupere le nouveau nom de dossier depuis le ficher csv
            dataset, initial_label, filename = path.split("/")

            # On récupere le reste du chemin et remplace le nom par le nouveau
            old_folder = f"{self.db_to_clean}/{dataset}/{initial_label}"
            new_folder = f"{self.db_to_clean}/{dataset}/{true_label}"

            if os.path.exists(old_folder):  # Si le dossier n'a pas encore été renommé
                try:
                    os.rename(old_folder, new_folder)  # On renomme le dossier
                except FileNotFoundError:
                    print(f"Le répértoire {old_folder} n'a pas été trouvé")

            return f"{dataset}/{true_label}/{filename}"

        # On remplace tous les noms de dossiers en fonction du csv précédemment modifié
        df["filepaths"] = df.apply(lambda row: adapt_file_path(row), axis=1)
        # endregion

        # Sauvegarde du csv final
        df.to_csv(f"{self.db_to_clean}/birds.csv", index=False)

    def correctString(self, string: str):
        """
        Renvoie la chaine de caractères sans espaces ni à droite ni à gauche, avec une majuscle au début
        et le reste en minuscule et sans apostroche
        """
        return string.strip().capitalize().replace("'", "")

    def full_correction(self):
        """
        Lance la correction complète :
        téléchargement du dataset, possible initialisation du dataset de test et correction
        """

        if not os.path.isfile(f"{self.db_to_clean}/birds_list.csv"):
            self.download_full_dataset()  # Si le fichier birds_list n'existe pas, on le télécharge
        if "ABBOTTS BABBLER" in os.listdir(f"{self.db_to_clean}/train"):
            self.dataset_correction()  # Si le premier dossier est en majuscule, alors on effectue les corrections
        if self.test_mode:
            self.test_phase_init()  # Si le mode test est activé, alors on réduit la taille du dataset


def main(db_to_clean: str = "./data/raw", test_mode: bool = False):
    dataset_correction = DatasetCorrection(db_to_clean=db_to_clean, test_mode=test_mode)
    dataset_correction.full_correction()


if __name__ == "__main__":
    main()
