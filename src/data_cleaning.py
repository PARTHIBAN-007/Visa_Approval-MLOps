import pandas as pd
import yaml
from datetime import date
from sklearn.preprocessing import  LabelEncoder
import joblib
class DataCleaning:
    def __init__(self):
        self.encoders = {}
        self.config = self.load_config()

    def load_config(self):
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            return config

        
    def clean_data(self, df: pd.DataFrame):
        
        # Drop irrelevant columns
        df = df.drop("case_id",axis=1)

        today = date.today()
        current_year = today.year
        df['company_age'] = current_year-df["yr_of_estab"]
        df = df.drop(columns="yr_of_estab")

        numeric_features = df.select_dtypes(exclude="object").columns
        categoric_features = df.select_dtypes(include="object").columns
        print(categoric_features)
            
        for columns in categoric_features:
            le = LabelEncoder()
            df[columns] = le.fit_transform(df[columns])
            self.encoders[columns] = le
                
        return df
    
    def register_encoder(self):
        encoder_path = self.config["encoder"]["encoder_path"]
        joblib.dump(self.encoders, encoder_path)
        print(f"Encoder saved to {encoder_path}")
    
    