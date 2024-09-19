import schedule
import time
from pipeline import run_pipeline
from app.utils.logger import setup_logger

logger = setup_logger("scheduler", "logs/scheduler.log")


def job():
    logger.info("Exécution planifiée de la pipeline...")
    try:
        run_pipeline()
        logger.info("Exécution de la pipeline terminée avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la pipeline : {str(e)}")


if __name__ == "__main__":
    logger.info("Configuration du planificateur...")
    schedule.every().day.at("12:00").do(job)  # Exécute la pipeline tous les jours à 12h00

    logger.info("Démarrage du planificateur...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Vérifie toutes les minutes
