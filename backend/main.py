import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os

app = FastAPI()

# Enable CORS for Next.js
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
MAPPINGS_PATH = os.path.join(BASE_DIR, 'model', 'mappings.pkl')
THRESHOLDS_PATH = os.path.join(BASE_DIR, 'model', 'thresholds.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'features.pkl')

print(f"Loading model from {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    mappings = joblib.load(MAPPINGS_PATH)
    thresholds = joblib.load(THRESHOLDS_PATH)
    try:
        model_features = joblib.load(FEATURES_PATH)
        print("Features list loaded successfully.")
    except FileNotFoundError:
        print("Warning: features.pkl not found. You need to run the notebook to generate it.")
        model_features = None
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model = None
    model_features = None

# Categorical columns list (from notebook)
CAT_COLS = [
    'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
    'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
    'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season'
]

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert features to DataFrame
        data = request.features
        df = pd.DataFrame([data])
        
        # Preprocessing
        # 1. Fill Missing Categories
        for c in CAT_COLS:
            if c in df.columns:
                df[c] = df[c].fillna('Missing')
                
                # 2. Apply Mapping
                if mappings and c in mappings:
                    mapping_dict = mappings[c]
                    
                    # Convert to string to ensure matching with mapping keys
                    # Map values and fill missing with -1
                    df[c] = df[c].astype(str).map(mapping_dict)
                    df[c] = df[c].fillna(-1).astype(int)
                else:
                    # If no mapping found but it's a categorical col, 
                    # we must convert it to int somehow or drop it to avoid str errors.
                    # For now, let's try to coerce or fill 0.
                    print(f"Warning: No mapping found for {c}")
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(int)
        
        # Ensure 'sii' is not in columns (it shouldn't be in request, but just in case)
        if 'sii' in df.columns:
            df = df.drop(columns=['sii'])
            
        # Align features to model training data
        if model_features is not None:
            # Add missing columns with 0
            for feature in model_features:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns to match training and drop extras
            df = df[model_features]
            
        # Make Prediction
        pred_raw = model.predict(df)
        
        # Apply Thresholds
        pred_rounded = threshold_Rounder(pred_raw, thresholds)
        
        return {
            "prediction_raw": float(pred_raw[0]),
            "prediction_class": int(pred_rounded[0])
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
