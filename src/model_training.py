import pandas as pd
import lightgbm as lgb

class ModelTraining:
    def __init__(self):
        self.lgbm = lgb.LGBMClassifier()
    
    def split_data(Self,df:pd.DataFrame):
        x = df.drop(columns="case_status")
        y = df["case_status"]
        return x, y
    
    def train_model(self,x_train:pd.DataFrame,y_train:pd.Series):
        self.lgbm.fit(x_train, y_train)
        return self.lgbm
    