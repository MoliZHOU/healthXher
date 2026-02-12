# healthXher Backend (FastAPI)

This is the core API for the healthXher system. It handles authentication, database persistence (via SQLCipher), and serves as the bridge to the Python ML inference engine.

## Features
- **FastAPI:** High-performance REST API.
- **SQLCipher:** Local-first, cryptographically secure SQLite database.
- **JWT Auth:** Bearer token authentication for secure sessions.
- **Pydantic:** Strict schema validation for all inputs/outputs.
- **Dependency Injection:** Modular architecture for database and security dependencies.

## Key Endpoints
- `POST /api/v1/auth/login`: Authenticate and receive a JWT.
- `POST /api/v1/predict`: Submit biometric data for RA risk analysis.
- `GET /api/v1/users/me`: Retrieve current user profile.

## Configuration
Configuration is managed in `app/core/config.py`. **IMPORTANT:** Sensitive keys must be set via environment variables:
- `SECRET_KEY`: Used for JWT signing.
- `DB_PASSPHRASE`: Encryption key for SQLCipher.
- `DEFAULT_ADMIN_PASSWORD`: (Optional) Set this to seed the `dadoudouoo` user on first run.

Example:
```bash
export SECRET_KEY="your-secret-hex"
export DB_PASSPHRASE="your-db-pass-hex"
```

## Development
To run the backend in development mode with hot-reload:
```bash
uvicorn app.main:app --reload --port 8000
```
Documentation is automatically generated and available at `/docs`.
