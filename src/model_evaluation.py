import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
class ModelEvaluation:
    def __init__(self):
        self.config = self.load_config()
        self.model = self.load_model()
    
    def load_config(self):
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config
    
    def load_model(self):
        model_path = self.config["model_path"]
        model = lgb.Booster(model_file=model_path)
        return model
    
    def evaluate_model(self,x_test:pd.DataFrame,y_test:pd.Series):
        y_pred = self.model.predict(x_test)
        acc = accuracy_score(y_test,y_pred)
        report = classification_report(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred)    
        print("Accuracy :",acc)
        print("Classification Report :\n",report)
        print("Confusion Matrix :\n",cm)
        
