import os
os.environ["DEFAULT_ADMIN_PASSWORD"] = "oodadoudou"
os.environ["SECRET_KEY"] = "test_secret_key"
os.environ["DB_PASSPHRASE"] = "test_db_pass"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_default_user_login():
    # Attempt to login with the seeded user
    response = client.post(
        "/api/v1/auth/login/",
        data={"username": "dadoudouoo", "password": "oodadoudou"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"
    print("\n[Success] Default user 'dadoudouoo' logged in successfully!")
