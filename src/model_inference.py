import pandas as pd
import yaml
import joblib
class ModelInference:
    def __init__(self):
        self.config = self.load_config()
        self.model = self.load_model()
        self.encoder = self.load_encoder()


    def load_config(self):
        with open("config.yml","r") as file:
            config = yaml.safe_load(file)
        return config
    
    def load_model(self):
        model_path = self.config["model"]["model_path"]
        model = joblib.load(model_path)
        return model
    
    def load_encoder(self):
        encoder_path = r"D:\Projects\ML projects\Visa Approval\model\encoder.pkl"

        encoder = joblib.load(encoder_path)
        return encoder

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
       
        for col in self.config["encoder"]["categorical_cols"]:
            if col in df.columns:
                df[col] = self.encoder.transform(df[col])
        predictions = self.model.predict(df)
        return predictions