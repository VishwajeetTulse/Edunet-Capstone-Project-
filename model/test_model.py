import pandas as pd
import numpy as np
import joblib
import os

# Load model artifacts
base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, 'model.pkl'))
thresholds = joblib.load(os.path.join(base_dir, 'thresholds.pkl'))
mappings = joblib.load(os.path.join(base_dir, 'mappings.pkl'))
feature_cols = joblib.load(os.path.join(base_dir, 'features.pkl'))

# Helper function to round predictions based on optimized thresholds
def threshold_rounder(pred, thresholds):
    # UPDATED for 3 Classes (Merge Moderate/Severe)
    return np.where(pred < thresholds[0], 0,
                    np.where(pred < thresholds[1], 1, 2))

# Define test cases for each severity level (0: None, 1: Mild, 2: Moderate, 3: Severe)
# These values are approximated based on domain knowledge of the features to likely trigger specific classes
test_cases = [
    {
        "name": "Case 0: Healthy/None",
        "data": {
            "Basic_Demos-Age": 20,
            "Basic_Demos-Sex": 1, # Male
            "Physical-BMI": 22.0,
            "Physical-Height": 175.0,
            "Physical-Weight": 70.0,
            "Physical-HeartRate": 70,
            "Physical-Diastolic_BP": 80,
            "Physical-Systolic_BP": 120,
            "Fitness_Endurance-Time_Mins": 10.0,
            "FGC-FGC_PU": 50.0,
            "BIA-BIA_Fat": 15.0,
            "PAQ_A-PAQ_A_Total": 3.0, # High Activity
            "PAQ_C-PAQ_C_Total": 3.0,
            "SDS-SDS_Total_Raw": 30, # Low Sleep Disturbance
            "PreInt_EduHx-computerinternet_hoursday": 1.0, # Low usage
            "CGAS-CGAS_Score": 85 # High functioning
        }
    },
    {
        "name": "Case 1: Mild",
        "data": {
            "Basic_Demos-Age": 18,
            "Basic_Demos-Sex": 0, # Female
            "Physical-BMI": 24.0,
            "Physical-Height": 165.0,
            "Physical-Weight": 65.0,
            "Physical-HeartRate": 75,
            "Physical-Diastolic_BP": 82,
            "Physical-Systolic_BP": 122,
            "Fitness_Endurance-Time_Mins": 6.0,
            "FGC-FGC_PU": 40.0,
            "BIA-BIA_Fat": 22.0,
            "PAQ_A-PAQ_A_Total": 2.5,
            "PAQ_C-PAQ_C_Total": 2.5,
            "SDS-SDS_Total_Raw": 45, # Moderate sleep issues
            "PreInt_EduHx-computerinternet_hoursday": 3.0,
            "CGAS-CGAS_Score": 70
        }
    },
    {
        "name": "Case 2: High Risk (Mod/Severe)",
        "data": {
            "Basic_Demos-Age": 16,
            "Basic_Demos-Sex": 1,
            "Physical-BMI": 28.0,
            "Physical-Height": 170.0,
            "Physical-Weight": 85.0,
            "Physical-HeartRate": 85,
            "Physical-Diastolic_BP": 85,
            "Physical-Systolic_BP": 130,
            "Fitness_Endurance-Time_Mins": 4.0,
            "FGC-FGC_PU": 25.0,
            "BIA-BIA_Fat": 28.0,
            "PAQ_A-PAQ_A_Total": 1.8, # Low Activity
            "PAQ_C-PAQ_C_Total": 1.8,
            "SDS-SDS_Total_Raw": 60, # High sleep disturbance
            "PreInt_EduHx-computerinternet_hoursday": 6.0, # High usage
            "CGAS-CGAS_Score": 60
        }
    },
    {
        "name": "Case 3: Extreme Risk (Should be Class 2)",
        "data": {
            "Basic_Demos-Age": 19,
            "Basic_Demos-Sex": 1,
            "Physical-BMI": 32.0,
            "Physical-Height": 170.0,
            "Physical-Weight": 95.0,
            "Physical-HeartRate": 95,
            "Physical-Diastolic_BP": 90,
            "Physical-Systolic_BP": 140,
            "Fitness_Endurance-Time_Mins": 2.0, # Very low endurance
            "FGC-FGC_PU": 10.0, # Poor coordination
            "BIA-BIA_Fat": 35.0,
            "PAQ_A-PAQ_A_Total": 1.2, # Sedentary
            "PAQ_C-PAQ_C_Total": 1.2,
            "SDS-SDS_Total_Raw": 75, # Severe sleep disturbance
            "PreInt_EduHx-computerinternet_hoursday": 12.0, # Extreme usage
            "CGAS-CGAS_Score": 40 # Low functioning
        }
    }
]

print("--- Testing Model Predictions ---")
print(f"Loaded Features ({len(feature_cols)}): {feature_cols[:5]} ...")

for case in test_cases:
    # Create DataFrame from minimal input
    input_dicts = case['data'].copy()
    
    # 1. Fill missing columns with defaults (Mean or Mode from training would be better, but using static defaults for test)
    # We loop through ALL expected model features
    for col in feature_cols:
        if col not in input_dicts:
            # Check if it's a categorical column we need to handle specially? 
            # For simplicity in this test script, we fill 0 or specific defaults
            if 'Season' in col:
                input_dicts[col] = 1 # Fall
            else:
                input_dicts[col] = 0 # Default numeric

    # Create DataFrame all at once to avoid fragmentation warning
    input_df = pd.DataFrame([input_dicts])[feature_cols]

    # 3. Predict raw score
    raw_pred = model.predict(input_df)[0]
    
    # 4. Apply thresholds
    final_class = threshold_rounder(raw_pred, thresholds)
    
    # 5. Interpret Class
    class_labels = {0: "None", 1: "Mild", 2: "High Risk (Mod/Severe)"}
    label = class_labels[int(final_class)]
    
    print(f"\n{case['name']}")
    print(f"  Inputs: Hours={case['data']['PreInt_EduHx-computerinternet_hoursday']}, BMI={case['data']['Physical-BMI']}, Sleep={case['data']['SDS-SDS_Total_Raw']}")
    print(f"  Raw Score: {raw_pred:.4f}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Predicted Class: {int(final_class)} ({label})")
