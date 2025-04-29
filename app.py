from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model_inference import ModelInference
import pandas as pd

# Initialize FastAPI app and model
app = FastAPI()
model = ModelInference()

# Pydantic input schema
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
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Call prediction method
        prediction = model.predict(df)

        # Return result
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
