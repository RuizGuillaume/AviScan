import unittest
import os
import pandas as pd
from datetime import datetime
from unittest.mock import patch
from monitoring.performance_tracker import PerformanceTracker


class TestPerformanceTracker(unittest.TestCase):
    @patch("monitoring.performance_tracker.DataManager")
    def setUp(self, mock_data_manager):
        self.test_log_file = f'logs/performance_logs_{datetime.now().strftime("%Y%m%d")}.csv'
        mock_data_manager.return_value.get_class_names.return_value = [
            "class1",
            "class2",
        ]
        self.tracker = PerformanceTracker()

    def tearDown(self):
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)

    def test_log_prediction(self):
        self.tracker.log_prediction("class1", 0.9, "class1")
        self.tracker.log_prediction("class2", 0.8, "class2")

        df = pd.read_csv(self.test_log_file)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["predicted_class"].tolist(), ["class1", "class2"])

    def test_get_performance_metrics(self):
        test_data = pd.DataFrame(
            {
                "date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 4,
                "predicted_class": ["class1", "class1", "class2", "class2"],
                "true_class": ["class1", "class2", "class2", "class2"],
                "confidence": [0.9, 0.8, 0.7, 0.9],
            }
        )
        test_data.to_csv(self.test_log_file, index=False)

        overall_accuracy, class_accuracies = self.tracker.get_performance_metrics(self.test_log_file)
        self.assertEqual(overall_accuracy, 0.75)
        self.assertEqual(class_accuracies, {"class1": 1.0, "class2": 2 / 3})


if __name__ == "__main__":
    unittest.main()
