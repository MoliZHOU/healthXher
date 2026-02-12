import os
os.environ["DEFAULT_ADMIN_PASSWORD"] = "SecurePassword123!"
os.environ["SECRET_KEY"] = "test_secret_key"
os.environ["DB_PASSPHRASE"] = "test_db_pass"

import pytest
from fastapi.testclient import TestClient
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app
from tests.utils.data_loader import SmartDataLoader

client = TestClient(app)

DATA_PATH = "data/The_final_data_after_screening.csv"

def test_system_lifecycle():
    # 1. DB Init check (Root endpoint)
    response = client.get("/")
    assert response.status_code == 200
    assert "healthXher" in response.json()["system"]

    # 2. User Registration
    username = "robot_user_001"
    password = "SecurePassword123!"
    reg_response = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password}
    )
    # Might already exist if rerun, handle gracefully
    assert reg_response.status_code in [200, 400]

    # 3. Login
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": username, "password": password}
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 4. Smart Data Loading
    loader = SmartDataLoader(DATA_PATH)
    payload, seqn = loader.get_random_patient()
    print(f"\n[Jimi Lekley QA] Simulating Patient ID: {seqn} from CSV...")

    # 5. Inference & Persistence
    pred_response = client.post(
        "/api/v1/predict/",
        json=payload,
        headers=headers
    )
    
    assert pred_response.status_code == 200
    data = pred_response.json()
    
    # Assertions
    assert 0.0 <= data["probability"] <= 1.0
    assert "risk_level" in data
    assert "derived_features" in data
    
    print(f"[Success] Probability: {data['probability']:.2f} | Risk: {data['risk_level']}")
    print(f"[Audit] Derived: {data['derived_features']}")

def test_age_edge_case():
    # Login again
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": "robot_user_001", "password": "SecurePassword123!"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Age 150 (Should fail Pydantic validation)
    payload = {
        "age": 150,
        "gender": "Female",
        "smoking_status": "Never",
        "fiber_consumption": 10.0
    }
    response = client.post("/api/v1/predict/", json=payload, headers=headers)
    assert response.status_code == 422
    print("[Success] Validation blocked Age: 150 correctly.")

def test_unauthorized_access():
    payload = {
        "age": 45,
        "gender": "Female",
        "waist_cm": 88.0,
        "height_cm": 165.0,
        "neutrophils": 4.5,
        "lymphocytes": 2.1,
        "smoking_status": "Never",
        "fiber_consumption": 15.0
    }
    response = client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 401
    print("[Success] Unauthorized access blocked.")

def test_invalid_login():
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "nonexistent", "password": "wrongpassword"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Incorrect username or password"
    print("[Success] Invalid login rejected.")

def test_biometric_boundaries():
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": "robot_user_001", "password": "SecurePassword123!"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Height too low
    bad_payload = {
        "age": 45,
        "gender": "Female",
        "waist_cm": 88.0,
        "height_cm": 10.0, # Boundary is 50
        "smoking_status": "Never",
        "fiber_consumption": 15.0
    }
    response = client.post("/api/v1/predict/", json=bad_payload, headers=headers)
    assert response.status_code == 422
    
    # Invalid Enum
    bad_payload["height_cm"] = 165.0
    bad_payload["gender"] = "InvalidOption"
    response = client.post("/api/v1/predict/", json=bad_payload, headers=headers)
    assert response.status_code == 422
    print("[Success] Biometric and Enum boundaries enforced.")

if __name__ == "__main__":
    # For manual run
    pytest.main([__file__])
