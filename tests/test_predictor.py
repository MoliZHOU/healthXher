import pytest
import pandas as pd
import numpy as np
from python.ra_pipeline import RAPredictor, FeatureEngineer, DOMWeighter, ExpertAugmentedGBDT

@pytest.fixture
def mock_predictor():
    engineer = FeatureEngineer()
    # Fit with some dummy data to initialize
    dummy_df = pd.DataFrame({
        "Age": [25, 60],
        "Gender": ["Male", "Female"],
        "SmokingStatus": ["Never", "Current"],
        "BMI": [22.0, 28.0],
        "FiberConsumption": [10.0, 20.0],
        "Waist_cm": [80.0, 95.0], # Note: Column names might vary in raw vs engineered
        "Height_cm": [175.0, 160.0],
        "RheumatoidArthritis": [0, 1],
        "BRI": [3.5, 5.0],
        "NLR": [2.0, 4.5],
        "DII": [1.5, 3.0]
    })
    # Initialize components
    engineer.fit_transform(dummy_df)
    weighter = DOMWeighter()
    
    # Create a dummy model that predicts 0.5 for everything for testing logic
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))
    
    return RAPredictor(DummyModel(), engineer, weighter)

def test_predictor_boundary_age(mock_predictor):
    # Test age boundaries
    df = pd.DataFrame([{
        "Age": 18,
        "Gender": "Male",
        "SmokingStatus": "Never",
        "BMI": 22.0,
        "FiberConsumption": 10.0,
        "BRI": 3.5,
        "NLR": 2.0,
        "DII": 1.5
    }])
    result = mock_predictor.predict(df)
    assert "RA_probability" in result.columns
    assert result["RA_risk_level"].iloc[0] == "High Risk (35-60%)" # Because 0.5 is in that bin

def test_predictor_missing_columns(mock_predictor):
    # Test behavior when columns are missing (FeatureEngineer should handle)
    df = pd.DataFrame([{
        "Age": 45,
        "Gender": "Female"
    }])
    # Should not crash
    result = mock_predictor.predict(df)
    assert len(result) == 1

def test_predictor_invalid_data_types(mock_predictor):
    # Test with string values in numeric columns
    df = pd.DataFrame([{
        "Age": "invalid",
        "Gender": "Male",
        "BMI": "not_a_number"
    }])
    # FeatureEngineer uses pd.to_numeric(errors='coerce').fillna(0.0)
    result = mock_predictor.predict(df)
    assert len(result) == 1
    assert result["RA_probability"].iloc[0] >= 0

def test_risk_level_bins(mock_predictor):
    # Verify the bins
    # RISK_BINS = [0, 0.15, 0.35, 0.60, 1.01]
    # 0.5 should be 'High Risk (35-60%)'
    df = pd.DataFrame([{"Age": 40}])
    result = mock_predictor.predict(df)
    assert result["RA_risk_level"].iloc[0] == "High Risk (35-60%)"

def test_explain_output(mock_predictor):
    row = pd.Series({"Age": 40, "Gender": "Male"})
    explanation = mock_predictor.explain(row)
    assert "predicted RA probability is 50.0%" in explanation
    assert "Note:" in explanation
