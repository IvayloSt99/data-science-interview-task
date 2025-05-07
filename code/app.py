from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

@app.get("/")
def root():
    return {"message": "ML pipeline API is live"}

@app.post("/predict/")
def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    with open("config/best_params_logistic.pkl", "rb") as f:
        best_params = pickle.load(f)
    # Load model with best_params, predict, return result...
    return {"input": input_data, "prediction": "placeholder"}