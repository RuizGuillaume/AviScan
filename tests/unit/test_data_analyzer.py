import unittest
import os
from app.utils.data_analyzer import DataAnalyzer


class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "class1"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "class2"), exist_ok=True)
        open(os.path.join(self.test_dir, "class1", "img1.jpg"), "w").close()
        open(os.path.join(self.test_dir, "class1", "img2.jpg"), "w").close()
        open(os.path.join(self.test_dir, "class2", "img1.jpg"), "w").close()

        self.analyzer = DataAnalyzer(self.test_dir)

    def tearDown(self):
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_get_class_counts(self):
        counts = self.analyzer.get_class_counts()
        self.assertEqual(counts, {"class1": 2, "class2": 1})

    def test_save_and_load_class_counts(self):
        test_file = "test_counts.json"
        self.analyzer.save_class_counts(test_file)
        loaded_counts = DataAnalyzer.load_class_counts(test_file)
        self.assertEqual(loaded_counts, {"class1": 2, "class2": 1})
        os.remove(test_file)


if __name__ == "__main__":
    unittest.main()
