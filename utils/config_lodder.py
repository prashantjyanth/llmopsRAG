import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def as_dict(self):
        return self.cfg
