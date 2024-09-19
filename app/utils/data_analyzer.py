import os
import json


class DataAnalyzer:
    def __init__(self, data_dir="data/train"):
        self.data_dir = data_dir

    def get_class_counts(self):
        class_counts = {}
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(
                    [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                )
        return class_counts

    def save_class_counts(self, output_file="initial_class_counts.json"):
        class_counts = self.get_class_counts()
        with open(output_file, "w") as f:
            json.dump(class_counts, f)

    @staticmethod
    def load_class_counts(input_file="initial_class_counts.json"):
        with open(input_file, "r") as f:
            return json.load(f)
