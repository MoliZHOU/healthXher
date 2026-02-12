from pydantic import BaseModel, Field
from typing import Optional, List

class DominantFeatureForm(BaseModel):
    age: int = Field(..., ge=18, le=120)
    gender: str = Field(..., pattern="^(Male|Female)$")
    waist_cm: Optional[float] = Field(None, gt=30, lt=200)
    height_cm: Optional[float] = Field(None, gt=50, lt=250)
    neutrophils: Optional[float] = Field(None, ge=0)
    lymphocytes: Optional[float] = Field(None, ge=0)
    smoking_status: str = Field(..., pattern="^(Never|Former|Current)$")
    bmi: Optional[float] = Field(None, gt=10, lt=100)
    fiber_consumption: float = Field(..., ge=0)
    dii: Optional[float] = None
    
    # Other potential fields from ra_pipeline
    physical_activity: str = "Sedentary"
    drinking_status: str = "Almost non-drinker"
    hypertension: str = "Normal"
    diabetes: str = "Normal"
    hyperlipidemia: str = "Normal"

class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    needs_followup: bool
    derived_features: dict
