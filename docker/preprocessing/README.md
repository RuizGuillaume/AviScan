# Preprocessing

Conteneur entièrement autonome, il est responsable de l'acquisition, du nettoyage et de l'augmentation des données, y compris les nouvelles images soumises par les utilisateurs.

## Composants

- `cleanDB.py`: Fonctions de netoyyage des datasets
- `DatasetCorrection.py`: Répare les incohérences du dataset Kaggle et génère optionnellement une version test du dataset
- `preprocessing.py`: Script de prétraitement du jeu de données, appelle tous les autres modules
- `SizeManager.py`: Vérifie et modifie la taille des images vers une résolution standardisée
- `UnderSampling.py`: Applique les fonctions de sous-échantillonnage aléatoire
