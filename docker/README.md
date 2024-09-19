# Docker

Ce dossier contient l'ensembles des conteneurs Docker du projet.

## Sommaire

- [Admin API](./admin_api) Conteneur de l'API administrative
- [Production](./inference) Conteneur du modèle en production
- [MLflow](./mlflowui) Conteneur de l'interface MLflow
- [Monitoring](./monitoring) Conteneur de surveillance de l'état de santé du système et du modèle en production 
- [Data processing](./preprocessing) Conteneur de traitement des données
- [Streamlit](./streamlit) Conteneur de l'interface Streamlit
- [Training](./training) Conteneur d'entraînement
- [User API](./user_api) Conteneur de l'API client

## Utilisation

- Pour démarrer les conteneurs Docker :
    - Si vous avez une carte graphique Nvidia : `docker-compose -f docker-compose-nvidia.yml up`.
    - Si vous n'avez pas de carte graphique Nvidia ou que vous n'êtes pas sûr : `docker-compose -f docker-compose.yml up`.