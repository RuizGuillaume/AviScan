import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.drift_monitor import DriftMonitor
from monitoring.alert_system import AlertSystem
from app.utils.logger import setup_logger

logger = setup_logger("drift_report", "logs/drift_report.log")


def main():
    drift_monitor = DriftMonitor()
    alert_system = AlertSystem()

    logger.info("Vérification du drift en cours...")
    drift_detected, reasons = drift_monitor.check_drift()

    if drift_detected:
        logger.warning(f"Drift détecté! Raisons : {reasons}")
        alert_system.send_alert(
            "Drift Détecté",
            f"Un drift a été détecté pour les raisons suivantes : {reasons}",
        )
    else:
        logger.info("Aucun drift détecté.")

    logger.info("Vérification terminée.")


if __name__ == "__main__":
    main()
