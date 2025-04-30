import pandas as pd
import yaml
import pickle
class ModelInference:
    def __init__(self):
        self.config = self.load_config()
        self.model,self.encoder = self.load_model()


    def load_config(self):
        with open("config.yml","r") as file:
            config = yaml.safe_load(file)
        return config
    
    def load_model(self):
       with open("./model/model1.pkl", "rb") as f:

        load_artifacts = pickle.load(f)

        model = load_artifacts['model']
        encoder = load_artifacts['label'] 
        print("Encoder")
        return model,encoder
        

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df)
        print(df.columns)
        for col in self.config["encoder"]["categorical_cols"]:
            if col in df.columns:
                df[col] = self.encoder[col].transform(df[col])
        print(df)
        print("__----------------------------------------------------------------------")
        predictions = self.model.predict(df)
        print(predictions)
        return predictions