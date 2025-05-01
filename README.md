#  Visa Approval ML System

This project is a full-stack machine learning application for predicting visa approval status using LightGBM. It includes:
- LightGBM Model is used to hanlde Class Imbalance in Data
-  SHAP for Model Interpretation
-  Model Training with MLflow Tracking
-  Model Serving via FastAPI
-  Interactive Frontend via Streamlit
-  Containerization with Docker
-  CI/CD with GitHub Actions
-  Deployment on AWS EC2 + ECR
--------------------------------------------------------------------------------------


## Tech Stack 
- **Programming** - Python(Pandas,Scikit-Learn)-3.12.4
- **ML Algorithm** - LightGBM
- **ML Interpreation** - SHAP
- **ML Model Tracking** - MLFlow
- **ML Model** - Joblib
- **Backend** - FastAPI
- **Frontend** - streamlit
- **CI/CD** - Github Actions
- **Containerization** - AWS ECR
- **Model Deployment** - AWS EC2(t2.medium)
--------------------------------------------------------------------------------------


##  Run Locally with Docker

### 1. Clone the Repo

```bash
git clone https://github.com/PARTHIBAN-007/Visa_Approval-MLOps.git
cd Visa_Approval-MLOps
```

### 2.Build Docker Iage
```bash
docker build -t visa-approval .
```

### 3.Run the Container
```bash
docker run -d -p 8501:8501 -p 8000:8000 -p 5000:5000 visa-approval
```


### 4. Access Services
```bash
Streamlit: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs
MLflow UI: http://localhost:5000
```
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------


##  Run Locally without Docker

### 1.Clone the repository

```bash
git clone https://github.com/PARTHIBAN-007/Visa_Approval-MLOps.git
cd Visa_Approval-MLOps
```
### 2.Create and activate virtual environment
```bash
python -m venv venv
sourcevenv\Scripts\activate 
```
### 3.Install dependencies
```bash
pip install -r requirements.txt
```
### 4.Data Ingestion, Cleaning and Training the Model(Mlflow Serving)
```bash
python main.py
```
### 5.Run MLflow tracking server 
```bash
mlflow ui
```
### 6.Start FastAPI server
```bash
uvicorn app:app --reload
```
### 7.Run Streamlit UI
```bash
streamlit run streamlit.py
```
