from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api import deps
from app.db.models import User, PredictionRecord
from app.schemas.predict import DominantFeatureForm, PredictionResponse
from app.services import calculator, inference

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
def predict(
    payload: DominantFeatureForm,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    # 1. Calculate Derived Features if missing
    bri = payload.dii # Temporary fallback logic if needed
    
    calc_bri = calculator.calculate_bri(payload.waist_cm, payload.height_cm)
    calc_nlr = calculator.calculate_nlr(payload.neutrophils, payload.lymphocytes)
    
    # Use provided BRI/NLR if present, else use calculated
    # For MVP, we prioritize backend calculation for "High-Assurance"
    final_bri = calc_bri if calc_bri > 0 else (0.0) # In real app, maybe payload has it
    final_nlr = calc_nlr if calc_nlr > 0 else (0.0)
    
    # DII Proxy
    final_dii = payload.dii if payload.dii is not None else calculator.estimate_dii_proxy(payload.fiber_consumption)

    # 2. Prepare features for model (Must match scripts/train_model.py FEATURES list)
    features = {
        "Age": payload.age,
        "Gender": payload.gender,
        "SmokingStatus": payload.smoking_status,
        "BMI": payload.bmi if payload.bmi else 25.0,
        "FiberConsumption": payload.fiber_consumption,
        "BRI": final_bri,
        "NLR": final_nlr,
        "DII": final_dii,
        "PhysicalActivity": payload.physical_activity,
        "Hypertension": payload.hypertension,
        "Diabetes": payload.diabetes
    }

    # 3. Inference
    probability = inference.predictor_service.predict(features)
    
    # 4. Determine Risk Level
    if probability >= 0.60:
        risk_level = "Very High Risk (>60%)"
    elif probability >= 0.35:
        risk_level = "High Risk (35-60%)"
    elif probability >= 0.15:
        risk_level = "Moderate Risk (15-35%)"
    else:
        risk_level = "Low Risk (<15%)"

    # 5. Persist to Encrypted DB
    record = PredictionRecord(
        user_id=current_user.id,
        input_data=payload.dict(),
        derived_features={"bri": final_bri, "nlr": final_nlr, "dii": final_dii},
        probability=probability,
        risk_level=risk_level
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "probability": probability,
        "risk_level": risk_level,
        "needs_followup": probability >= 0.15,
        "derived_features": record.derived_features
    }
