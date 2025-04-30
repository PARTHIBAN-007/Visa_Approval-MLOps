import os
import yaml
import pickle
class ModelRegistry:
    def __init__(self):
        self.config = self.load_config()
        self.model = None

    def load_config(self):
        with open("config.yml","r") as file:
            config = yaml.safe_load(file)
        return config
    
    # def register_encoder(self,encoder):
    #     encoder_path = self.config["encoder"]["encoder_path"]
    #     joblib.dump(encoder, encoder_path)
    
    def register_model(self,model,label):
        artifacts  = {
            'model':model,
            'label':label
        }
        with open("./model/model1.pkl", 'wb') as file:
            pickle.dump(artifacts, file)


        