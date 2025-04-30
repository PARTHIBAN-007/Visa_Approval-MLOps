import pandas as pd
import yaml
import joblib
class ModelInference:
    def __init__(self):
        self.config = self.load_config()
        self.model,self.encoder = self.load_model()


    def load_config(self):
        with open("config.yml","r") as file:
            config = yaml.safe_load(file)
        return config
    
    def load_model(self):
        model_path = self.config["model"]["model_path"]
        encoder_path = self.config["encoder"]["encoder_path"]
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model,encoder

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        
        for col in self.config["encoder"]["categorical_cols"]:
            if col in df.columns:
                df[col] = self.encoder[col].transform(df[col])
        predictions = self.model.predict(df)
        return predictions