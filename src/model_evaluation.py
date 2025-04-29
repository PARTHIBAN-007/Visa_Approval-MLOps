import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
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
        with mlflow.start_run():
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_params(self.config["model_params"])
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", report["f1-score"])
            mlflow.log_metric("precision", report["precision"])
            mlflow.log_metric("recall", report["recall"])
            mlflow.log_metric("confusion_matrix", cm)   
        
            
        
        