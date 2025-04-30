from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model_inference import ModelInference
import pandas as pd

app = FastAPI()
model = ModelInference()

class InputData(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    region_of_employment: str
    unit_of_wage: str
    full_time_position: str
    no_of_employees: int
    prevailing_wage: float
    company_age: int

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    prediction = model.predict(df)
 
    return {"prediction":str(prediction[0])}