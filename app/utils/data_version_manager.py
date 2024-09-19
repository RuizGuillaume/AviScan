import os
import json
from datetime import datetime


class DataVersionManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.version_file = os.path.join(data_dir, "data_version.json")

    def get_current_version(self):
        if os.path.exists(self.version_file):
            with open(self.version_file, "r") as f:
                version_info = json.load(f)
            return version_info["version"]
        return None

    def update_version(self):
        new_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_info = {"version": new_version, "timestamp": datetime.now().isoformat()}
        with open(self.version_file, "w") as f:
            json.dump(version_info, f)
        return new_version
