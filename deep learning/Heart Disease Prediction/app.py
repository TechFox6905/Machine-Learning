from fastapi import FastAPI, Depends
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import Patient

# Initialize FastAPI
app = FastAPI()

# Load Model and Scaler
model = tf.keras.models.load_model("heart_disease_model.h5")
scaler = joblib.load("scaler.pkl")

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Model for Input Data
class PatientData(BaseModel):
    age: int
    gender: int
    cholesterol: int
    blood_pressure: int
    heart_rate: int
    glucose: int

@app.post("/predict/")
def predict_heart_disease(patient: PatientData, db: Session = Depends(get_db)):
    # Convert input to NumPy array
    data = np.array([[patient.age, patient.gender, patient.cholesterol, patient.blood_pressure, patient.heart_rate, patient.glucose]])

    # Apply the same scaling as training
    data = scaler.transform(data)

    # Make Prediction
    prediction = model.predict(data)
    result = int(prediction[0][0] > 0.5)  # Convert probability to binary outcome

    # Store Result in Database
    new_patient = Patient(**patient.dict(), outcome=result)
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return {"patient_id": new_patient.id, "prediction": result}
