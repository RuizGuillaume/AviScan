import os
import threading
import time
import queue
import logging
from datetime import datetime
from alert_system import AlertSystem
from system_monitor import SystemMonitor


# region Configuration
volume_path = 'volume_data'
log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_folder, "monitoring.log"), level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p')

alert_system = AlertSystem()
# endregion


class SystemMonitorThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.monitor = SystemMonitor()
        self.metrics_queue = queue.Queue()
        self.last_email = 0

    def run(self):
        """
        Démarre le thread de surveillance des performances de la machine host
        """
        logging.info("Démarrage du thread de monitoring de performances systèmes.")
        while not self.stop_event.is_set():
            metrics = self.monitor.get_metrics()  # Récupérer les métriques systèmes
            self.metrics_queue.put((time.time(), metrics))  # Les ajouter dans la file
            self.log_metrics()  # Les sauvegarder dans le csv de suivi des métriques
            self.check_metrics_consistency(metrics)  # Vérifier qu'elles n'indiquent pas de surcharge du système
            time.sleep(5)

    def stop(self):
        """
        Arrête le thread de surveillance des performances de la machine host
        """
        logging.info("Arrêt du thread de monitoring de performances systèmes.")
        self.stop_event.set()

    def log_metrics(self):
        """
        Enregistre les performances de la machine host dans les logs
        """
        while not self.metrics_queue.empty():
            timestamp, metrics = self.metrics_queue.get()
            self.monitor.log_metrics(metrics, datetime.fromtimestamp(timestamp))

    def check_metrics_consistency(self, metrics: dict):
        """
        Vérifie que les performances ne dépassent pas les seuils enregistrés, sinon envoie un email d'alerte
        """
        conditions = [
            (metrics["cpu_usage"] >= 99, "L'usage du CPU dépasse les 99%."),
            (metrics["memory_usage"] >= 80, "L'usage de la RAM dépasse les 80%."),
            (metrics["disk_usage"] >= 60, "Le stockage utilisé dépasse les 60%."),
            (metrics["swap_usage"] >= 50, "L'usage du SWAP dépasse les 50%."),
            (metrics["process_count"] >= 1000, "Le nombre de processus en cours est supérieur à 1000."),
        ]
        i = 0
        while f"gpu_{i}_usage" in metrics:
            conditions.append((metrics[f"gpu_{i}_usage"] >= 99, f"L'usage du GPU {i} dépasse les 99%."))
            i += 1

        for condition, message in conditions:
            if condition:
                self.send_alert_email(message)

    def send_alert_email(self, message):
        """
        Envoie un email d'alerte
        """
        # On envoie pas plus d'un email d'alerte par heure
        current_time = time.time()
        if current_time - self.last_email >= 3600:
            logging.error("Anomalie de performances, email de rapport envoyé.")
            subject = "Anomalie de permances"
            alert_system.send_alert(subject=subject, message=message)
            self.last_email = current_time


if __name__ == "__main__":
    try:
        monitor_thread = SystemMonitorThread()
        monitor_thread.run()
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution du suivi des métriques système: {e}")
        alert_system.send_alert(subject="Erreur lors du suivi des métriques système",
                                message=f"Erreur lors de l'exécution du suivi des métriques système: {e}")
