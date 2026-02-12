import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder

# --- Configuration & Constraints ---
DATA_PATH = "data/The_final_data_after_screening.csv"
MODEL_PATH = "app/core/model_cache.pkl"
FEATURES_PATH = "app/core/model_features.json"

# Features selected for the "Dominant Form" (Must match Backend API output)
FEATURES = [
    "Age", "Gender", "SmokingStatus", "BMI", 
    "FiberConsumption", "BRI", "NLR", "DII",
    "PhysicalActivity", "Hypertension", "Diabetes"
]

# Monotonic Constraints: 1 (Increase), -1 (Decrease), 0 (None)
# Age up -> Risk up, BRI up -> Risk up, Fiber up -> Risk down
MONOTONIC_MAP = {
    "Age": 1,
    "BRI": 1,
    "FiberConsumption": -1,
    "NLR": 1,
    "DII": 1
}

def train():
    print(f"[Jimi Lekley] Starting Training Protocol on {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing training data at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # Simple feature engineering for training data (ensure NLR exists)
    if 'NLR' not in df.columns and 'LBDNENO' in df.columns:
        df['NLR'] = df['LBDNENO'] / df['LBDLYMNO'].replace(0, np.nan)
        df['NLR'] = df['NLR'].fillna(df['NLR'].median())
    elif 'NLR' not in df.columns:
        df['NLR'] = 2.0 # Fallback median

    if 'DII' not in df.columns:
        df['DII'] = 0.0 # Fallback

    # Target
    y = df["RheumatoidArthritis"]
    
    # Filter features and handle categoricals
    X = df[FEATURES].copy()
    
    # Encode categoricals for XGBoost
    cat_encoders = {}
    for col in ["Gender", "SmokingStatus", "PhysicalActivity", "Hypertension", "Diabetes"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        cat_encoders[col] = le
    
    # Prepare Monotonic Constraints list in feature order
    mono_constraints = tuple([MONOTONIC_MAP.get(f, 0) for f in FEATURES])

    # Model parameters
    params = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "monotonic_constraints": mono_constraints,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X, y)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(FEATURES, f)
    
    print(f"[Success] Model saved to {MODEL_PATH}")
    print(f"[Success] Feature schema saved to {FEATURES_PATH}")
    print(f"Columns: {FEATURES}")
    print(f"Constraints: {mono_constraints}")

if __name__ == "__main__":
    train()
