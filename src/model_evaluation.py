import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score,f1_score
import mlflow
import mlflow.lightgbm
import joblib
class ModelEvaluation:
    def __init__(self):
        self.config = self.load_config()
        self.model = self.load_model()
    
    def load_config(self):
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
        return config
    
    def load_model(self):
        model_path = self.config["model"]["model_path"]
        model = joblib.load(model_path)
        return model
    
    def evaluate_model(self,x_test:pd.DataFrame,y_test:pd.Series):
        y_pred = self.model.predict(x_test)
        acc = accuracy_score(y_test,y_pred)
        report = classification_report(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred)  
        mlflow.set_experiment("Visa Approval {}")
        with mlflow.start_run():
            mlflow.lightgbm.log_model(self.model, artifact_path="model")
            mlflow.log_params(self.config["model_params"])
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1_score(y_test,y_pred))
            mlflow.log_metric("precision", precision_score(y_test,y_pred))
            mlflow.log_metric("recall", recall_score(y_test,y_pred))
        
            
        
        