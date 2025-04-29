import pandas as pd
import yaml
from datetime import date
from sklearn.preprocessing import  LabelEncoder
import joblib
class DataCleaning:
    def __init__(self):

        self.encoder = LabelEncoder()

        
    def clean_data(self, df: pd.DataFrame):
        
        # Drop irrelevant columns
        df = df.drop("case_id",axis=1)

        today = date.today()
        current_year = today.year
        df['company_age'] = current_year-df["yr_of_estab"]
        df = df.drop(columns="yr_of_estab")

        numeric_features = df.select_dtypes(exclude="object").columns
        categoric_features = df.select_dtypes(include="object").columns
            
        for columns in categoric_features:
            df[columns] = self.encoder.fit_transform(df[columns])
        joblib.dump(self.encoder, "model/encoder.pkl")
                      
        return df
    
    