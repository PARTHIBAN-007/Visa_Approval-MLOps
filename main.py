from src.data_ingestion import DataIngestion
from src.data_cleaning import DataCleaning
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation
from sklearn.model_selection import train_test_split

def main():
    # Data Ingestion
    data_ingestion = DataIngestion()
    df = data_ingestion.load_data()
    print("Data Ingestion Completed")
    # Data Cleaning
    data_cleaning = DataCleaning()  
    cleaned_data = data_cleaning.clean_data(df)
    data_cleaning.register_encoder()
    print("Data Cleaning Completed")

    feature_engineering  = FeatureEngineering()
    x, y = feature_engineering.split_data(cleaned_data)
    print("Feature Engineering Completed")

    x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=42)    
    # Model Training
    model_training = ModelTraining()
    model = model_training.train_model(x_train, y_train)
    model_training.register_model(model)
    print("Model Training Completed")

    # Model Evaluation
    model_evaluation = ModelEvaluation()
    model_evaluation.evaluate_model(x_test, y_test)
    print("Model Evaluation Completed")
   

if __name__=="__main__":
    main()