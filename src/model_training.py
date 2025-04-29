import pandas as pd
import lightgbm as lgb
import yaml
class ModelTraining:
    def __init__(self):
        self.config = self.load_config()
        self.model_params = self.config["model_params"]
        self.lgbm = lgb.LGBMClassifier(**self.model_params)

    def load_config(self):
        with open("config.yml","r") as file:
            config = yaml.safe_load(file)
        return config
    def split_data(Self,df:pd.DataFrame):
        x = df.drop(columns="case_status")
        y = df["case_status"]
        return x, y
    
    def train_model(self,x_train:pd.DataFrame,y_train:pd.Series):
        self.lgbm.fit(x_train, y_train)
        return self.lgbm
    