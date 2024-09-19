## Preprocessing

Ce dossier contient les scripts pour les prétraitements obligatoires des données avant la phase d'entraînement.

## Composants

- `preprocess_dataset.py`: Script de prétraitement du jeu de données, appelle les modules ci-dessous
- `DatasetCorrection.py`: Répare les incohérences du dataset Kaggle et génère optionnellement une version test du dataset
- `SizeManager.py`: Vérifie et modifie la taille des images vers une résolution standardisée
- `UnderSampling.py`: Applique les fonctions de sous-échantillonnage aléatoire

## Utilisation

Ces modules sont utilisés par la pipeline principale pour s'assurer de la cohérence et de la qualité des données.