# Monitoring

Le conteneur de monitoring, surveille en temps réel les performances de la machine et réévalue le comportement du modèle en production avec les nouvelles données pour alerter d'un potentiel drift.

## Composants

- `alert_system.py`: Gère l'envoi d'alertes en cas de problèmes détectés
- `drift_monitor.py`: Détecte les dérives du modèle en production
- `monitor.py`: Suit et enregistre les performances de la machine
- `system_monitor.py`: Recueille les différentes informations renseignant sur l'état de la machine
