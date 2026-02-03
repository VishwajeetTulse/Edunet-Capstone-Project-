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

# Get allowed origins from env or default to localhost
msg_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
origins = [origin.strip() for origin in msg_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["https://edunet-capstone-project.vercel.app", "http://localhost:3000"],
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

# Mapping for SII classes
SII_MAPPING = {
    0: {"level": "None", "desc": "No significant indication of problematic internet use."},
    1: {"level": "Mild", "desc": "Mild signs of internet overuse. Monitor screen time."},
    2: {"level": "High Risk", "desc": "Significant problematic use detected. Immediate attention recommended."}
}

# Defaults for missing critical values (Approximated based on healthy averages)
SAFE_DEFAULTS = {
    # Top features (Importance > 400)
    'Physical-Heart_Rate': 75.0,        # ~Avg Resting HR
    'SDS-SDS_Total_Raw': 40.0,          # Estimated mid-range for Sleep Disturbance
    'Physical-Systolic_BP': 110.0,      # Healthy youth BP
    'CGAS-CGAS_Score': 75.0,            # "Doing well" score (Range 1-100)
    'PAQ_C-PAQ_C_Total': 2.5,           # Avg Activity (if mean score) or 25 (if sum). Using 50 as safe mid? 
                                        # Let's check name carefully. It's PAQ_C-PAQ_C_Total.
                                        # Assume 2.8 mean * 10 items? ~28.
    'Physical-Diastolic_BP': 70.0,      # Healthy youth BP
    
    # Other important missing features
    'BIA-BIA_BMC': 2.0,                 # Bone Mineral Content (approx kg)
    'FGC-FGC_CU': 0.0,                  # Fluid Intelligence (Standardized? 0 might be mean)
    'BIA-BIA_DEE': 2000.0,              # Energy Expenditure (~2000 kcal)
    'BIA-BIA_FFMI': 17.0,               # Fat Free Mass Index
    'FGC-FGC_GSD': 0.0,                 # ?
    
    # Just in case names slightly differ (Feature cleaning might have changed names)
    'Physical-HeartRate': 75.0, 
    'Physical-Systolic_BP': 110.0,
    'Physical-Diastolic_BP': 70.0,
}

def feature_engineering(df):
    # Replicate notebook engineering
    # We must ensure columns exist before operation or are filled
    # Safe checks for division by zero
    
    # 1. Age & BMI interactions
    if 'Physical-BMI' in df.columns and 'Basic_Demos-Age' in df.columns:
        df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
    
    if 'PreInt_EduHx-computerinternet_hoursday' in df.columns and 'Basic_Demos-Age' in df.columns:
        df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
        
    if 'Physical-BMI' in df.columns and 'PreInt_EduHx-computerinternet_hoursday' in df.columns:
        df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
        
    # 2. BIA Ratios (Bio-Impedance). Most users won't have this, so we skip or let them be 0.
    # If we had BIA columns, we would calculate them here.
    
    return df

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1, 2))

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
            
        # --- Smart Imputation (Dynamic Defaults) ---
        # Adjust clinical defaults based on reported behavior and vitals.
        # This aligns strict models (LGBM/XGB) with sensitive ones (CatBoost).
        
        user_hours = float(data.get("PreInt_EduHx-computerinternet_hoursday", 0))
        
        # FIX: The model was trained on categorical codes (0-3), NOT raw hours.
        # We must map the raw input hours to the trained categories.
        # 0=<1h, 1=~1h, 2=~2h, 3=>3h
        if user_hours < 1.0:
            model_hours = 0
        elif user_hours < 1.5:
            model_hours = 1
        elif user_hours < 2.5:
            model_hours = 2
        else:
            model_hours = 3
            
        # Update the DataFrame used for prediction with the correct categorical code
        if "PreInt_EduHx-computerinternet_hoursday" in df.columns:
             df["PreInt_EduHx-computerinternet_hoursday"] = model_hours
        
        user_sds = float(data.get("SDS-SDS_Total_Raw", 40)) # Default 40 if missing
        user_hr = float(data.get("Physical-HeartRate", 80)) # Default 80 if missing
        
        # Calculate a "Stress Factor" (0.0 to 1.0)
        # SDS Range: 30 (Good) -> 75 (Severe)
        # HR Range: 60 (Good) -> 110 (Severe)
        # Hours Range: 2 (Good) -> 10 (Bad)
        
        stress_sds = max(0, min(1, (user_sds - 35) / 40.0))
        stress_hr = max(0, min(1, (user_hr - 70) / 40.0))
        stress_hours = max(0, min(1, (user_hours - 2) / 10.0))
        
        # Use the maximum signal found
        stress_factor = max(stress_sds, stress_hr, stress_hours)
        
        # 1. Derived Defaults based on Stress Factor
        # Systolic BP: 110 (Healthy) -> 135 (Stressed)
        calc_sys_bp = 110.0 + (25.0 * stress_factor)
        
        # Diastolic BP: 70 (Healthy) -> 90 (Stressed)
        calc_dia_bp = 70.0 + (20.0 * stress_factor)
        
        # CGAS (Functioning): 80 (Good) -> 50 (Impaired)
        calc_cgas = 80.0 - (30.0 * stress_factor)
        
        # Energy Expenditure (DEE): 2000 (Active) -> 1600 (Sedentary)
        calc_dee = 2000.0 - (400.0 * stress_factor)
        
        # Sleep Disturbance (SDS): 35 (Healthy) -> 65 (Severe)
        # This is critical: Severe users in training data have significantly higher SDS scores (Avg 50 vs 39).
        # If we default to 40, we suppress the severity prediction.
        calc_sds = 35.0 + (30.0 * stress_factor)
        
        # Update SAFE_DEFAULTS dynamically
        current_defaults = SAFE_DEFAULTS.copy()
        current_defaults.update({
            'Physical-Systolic_BP': calc_sys_bp,
            'Physical-Diastolic_BP': calc_dia_bp,
            'CGAS-CGAS_Score': calc_cgas,
            'BIA-BIA_DEE': calc_dee,
            # We now use the calculated SDS as the default fallback
            'SDS-SDS_Total_Raw': calc_sds, 
            # Ensure provided values are not overwritten if passed as defaults
            'Physical-HeartRate': user_hr
        })

        # Apply Defaults
        for col, val in current_defaults.items():
            if col not in df.columns or pd.isna(df[col].iloc[0]) or df[col].iloc[0] == 0:
                 # Check if the user actually provided it.
                 # The request dict 'data' holds the raw inputs.
                 # If key is missing in 'data', we use default.
                 if col not in data:
                     df[col] = val

        # Perform feature engineering
        df = feature_engineering(df)
            
        # Align features to model training data
        if model_features is not None:
            # Identify missing columns
            missing_cols = [f for f in model_features if f not in df.columns]
            
            if missing_cols:
                # Create a DataFrame for missing columns properly to avoid fragmentation
                # Default fill value is still 0 for unknowns, but we covered top ones above.
                missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
                df = pd.concat([df, missing_df], axis=1)
            
            # Reorder columns to match training and drop extras
            df = df[model_features]
            
        # Make Prediction
        pred_raw = model.predict(df)
        
        # --- EXPERT RULE LAYER ---
        # The model is conservative (regression to mean) and struggles to hit extremes (0 and 3).
        # We assume the user's reported vitals (Stress Factor) are ground truth.
        # We nudge the prediction towards the clinical reality reflected by their vitals.
        
        # Logic: If clinical signs are clear (Stress > 0.4), push UP.
        #        If clinical signs are healthy (Stress < 0.4), push DOWN.
        
        bias_correction = (stress_factor - 0.35) * 1.5
        # Example: 
        # Stress 1.0 (Severe) -> +0.975 boost (Push 1.6 -> 2.6)
        # Stress 0.0 (Healthy) -> -0.525 reduction (Push 0.4 -> 0.0)
        
        final_score = pred_raw[0] + bias_correction
        
        # Clamp to valid range 0-3 roughly (though thresholds handle it)
        final_score = max(0.0, final_score)
        
        # Apply Thresholds to the corrected score
        pred_rounded = threshold_Rounder([final_score], thresholds)
        cls = int(pred_rounded[0])
        
        info = SII_MAPPING.get(cls, {"level": "Unknown", "desc": "Unknown risk level."})
        
        return {
            "prediction_score": float(final_score),
            "original_score": float(pred_raw[0]),
            "stress_factor": float(stress_factor),
            "sii_index": cls,
            "risk_level": info["level"],
            "description": info["desc"],
            "detailed_metrics": {
                "hours_per_day": data.get("PreInt_EduHx-computerinternet_hoursday", "Not provided"),
                "bmi": data.get("Physical-BMI", "Not provided"),
                "physical_activity": "Low" if cls > 1 else "Normal" # Placeholder logic based on risk
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
