import unittest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.pipeline import run_pipeline


class TestPipeline(unittest.TestCase):
    @patch("scripts.pipeline.mlflow")
    @patch("scripts.pipeline.train_model")
    @patch("scripts.pipeline.DataManager")
    @patch("scripts.pipeline.DriftMonitor")
    @patch("scripts.pipeline.PerformanceTracker")
    @patch("scripts.pipeline.AlertSystem")
    @patch("scripts.pipeline.predictClass")
    @patch("scripts.pipeline.preprocess_data")
    def test_run_pipeline(
        self,
        mock_preprocess_data,
        mock_predictClass,
        mock_AlertSystem,
        mock_PerformanceTracker,
        mock_DriftMonitor,
        mock_DataManager,
        mock_train_model,
        mock_mlflow,
    ):
        # Configuration des mocks
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "1"
        mock_train_model.return_value = (MagicMock(), False)

        mock_data_manager = MagicMock()
        mock_data_manager.load_new_data.return_value = [
            ("/path/to/image1.jpg", "class1"),
            ("/path/to/image2.jpg", "class2"),
        ]
        mock_data_manager.get_class_names.return_value = ["class1", "class2"]
        mock_DataManager.return_value = mock_data_manager

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("class1", 0.9)
        mock_predictClass.return_value = mock_predictor

        mock_performance_tracker = MagicMock()
        mock_performance_tracker.get_performance_metrics.return_value = (
            0.85,
            {"class1": 0.9, "class2": 0.8},
        )
        mock_PerformanceTracker.return_value = mock_performance_tracker

        mock_drift_monitor = MagicMock()
        mock_drift_monitor.check_drift.return_value = (False, "No drift detected")
        mock_DriftMonitor.return_value = mock_drift_monitor

        # Mock for preprocess_data
        mock_preprocess_data.return_value = "mocked_data_version"

        # Exécution de la pipeline
        run_pipeline(test_dataset_mode=True)

        # Vérifications
        mock_mlflow.create_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        mock_train_model.assert_called_once()
        mock_data_manager.load_new_data.assert_called_once()
        mock_predictor.predict.assert_called()
        mock_performance_tracker.get_performance_metrics.assert_called_once()
        mock_drift_monitor.check_drift.assert_called_once()
        mock_preprocess_data.assert_called_once()

        # Vérifiez que les métriques importantes sont enregistrées
        mock_mlflow.log_metric.assert_any_call("total_images_processed", 2)
        mock_mlflow.log_metric.assert_any_call("overall_accuracy", 0.85)

        # Vérifiez que les paramètres importants sont enregistrés
        mock_mlflow.log_param.assert_any_call("drift_detected", False)

        print("Test de la pipeline réussi!")


if __name__ == "__main__":
    unittest.main()
