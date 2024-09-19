# Production

Le conteneur de production héberge le modèle optimisé pour des prédictions en temps réel sur les nouvelles images soumises.
Il peut être mis à jour régulièrement pour intégrer les améliorations basées sur les contributions des utilisateurs.

## Composants

- `alert_system.py`: Gère l'envoi d'alertes en cas de problèmes détectés
- `inference.py`: Détecte les dérives du modèle en production
