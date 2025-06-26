from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("models/chewy_recommender_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

@app.get("/")
def read_root():
    return {"status": "Chewy Recommender API is live 🐶"}

@app.post("/predict/")
def predict(breed: str, weight: float, height: float, symptom: str):
    try:
        breed_enc = encoder.transform([breed])[0]
    except:
        return {"error": f"Breed '{breed}' not found in training data."}
    
    symptom_enc = 0  # fallback
    for s in encoder.classes_:
        if symptom.lower().split()[0] in s.lower():
            symptom_enc = encoder.transform([s])[0]
            break

    input_df = pd.DataFrame([[breed_enc, weight, height, symptom_enc]],
                            columns=["breed_enc", "weight", "height", "symptom_enc"])
    pred = model.predict(input_df)[0]
    category = encoder.inverse_transform([pred])[0]
    return {"predicted_category": category}
