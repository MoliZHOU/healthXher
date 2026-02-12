# healthXher: Python ML Pipeline

This directory contains the "Brain" of the system—the machine learning models and inference logic for Rheumatoid Arthritis prediction.

## Core Components
- `ra_pipeline.py`: The main entry point for the ML pipeline. It handles feature mapping, normalization, and inference.
- `model_cache.pkl`: The serialized pre-trained XGBoost model.
- `model_features.json`: Metadata defining the feature names and expected types.
- `interface.py`: A high-level interface for integrating the ML model into the FastAPI application.

## Feature Engineering: "BioMatic Logic"
The pipeline doesn't just pass raw data to the model. it implements "BioMatic Logic"—biological constraints and derived features such as:
- **BMI:** Calculated from Height and Weight.
- **NLR (Neutrophil-to-Lymphocyte Ratio):** An inflammatory marker derived from blood counts.
- **BRI (Body Roundness Index):** A more precise metric for central obesity than BMI.

## Training
The training scripts are located in `scripts/train_model.py`. Training is performed on NHANES datasets, focusing on minimizing False Negatives in clinical settings.

## Usage
```python
from python.interface import RAPredictor

predictor = RAPredictor()
result = predictor.predict({
    "age": 45,
    "gender": "Female",
    "waist_cm": 88.5,
    # ... other features
})
print(result)
```
