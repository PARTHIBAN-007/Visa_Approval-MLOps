import joblib
import os
import yaml
class ModelRegistry:
    def __init__(self):
        self.config = self.load_config()
        self.model = None

    def load_config(self):
        with open("config.yml","r") as file:
            config = yaml.safe_load(file)
        return config
    def register_model(self,model):
        model_path = self.config["model"]["model_path"]
        if not os.path.exists(model_path):
            os.makedirs(model_path)
       
        joblib.dump(model, model_path)

        