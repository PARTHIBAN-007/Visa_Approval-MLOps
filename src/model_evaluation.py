import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score,f1_score
import mlflow
import mlflow.lightgbm
import joblib
from mlflow.models.signature import infer_signature


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Visa Approval-MLOps")

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
    
    def evaluate_model(self, x_test: pd.DataFrame, y_test: pd.Series):
        y_pred = self.model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        with mlflow.start_run():
            input_example = x_test.iloc[:1]
            signature = infer_signature(x_test, y_pred)

            mlflow.lightgbm.log_model(
                lgb_model=self.model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            mlflow.log_params(self.config["model_params"])

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
