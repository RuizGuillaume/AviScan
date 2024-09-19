# Entraînement

Ce conteneur orchestre l'entraînement du modèle, s'adaptant aux nouvelles classes et données. Il utilise MLflow pour le suivi des expériences et la gestion des versions, crucial vis à vis de l'évolution constante du dataset.

## Composants

- `alert_system.py`: Gère l'envoi d'email de rapport d'entraînement
- `training.py`: Script d'entraînement
