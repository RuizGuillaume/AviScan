import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime, timedelta


def setup_logger(name, log_file, level=logging.INFO, max_bytes=10 * 1024 * 1024, backup_count=5):
    """Function to setup as many loggers as you want"""

    # Créer le dossier logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)

    # Utiliser un seul fichier de log par jour
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f"logs/{name}_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def clean_old_logs(log_dir="logs", days_to_keep=30):
    """Supprime les fichiers de logs plus anciens que le nombre de jours spécifié"""
    now = datetime.now()
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_time > timedelta(days=days_to_keep):
                os.remove(file_path)
                print(f"Suppression du fichier de log ancien : {file_path}")
