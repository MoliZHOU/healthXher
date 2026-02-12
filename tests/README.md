# healthXher Testing Procedures

Quality assurance and clinical validation are critical. This directory contains the test suite for the healthXher system.

## Test Categories

### 1. Backend Integration Tests (`test_lifecycle.py`)
Verifies the full API lifecycle:
- User registration and login.
- Database encryption verification.
- Protected endpoint access.
- Correct API response mapping.

### 2. Model Verification
Tests the `RAPredictor` interface to ensure:
- Correct handling of edge cases (e.g., minimum/maximum age).
- Accurate feature derivation (BMI, NLR, BRI).
- Consistency of probability scores.

### 3. Data Loaders (`utils/data_loader.py`)
Tests for verifying that training and testing data are loaded correctly and match the expected schemas.

## Running Tests

We use `pytest` as our primary test runner.

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_lifecycle.py
```

## Continuous Integration
All tests must pass before code is merged into the main branch. The backend tests require a clean environment as they initialize a temporary SQLite/SQLCipher database for isolation.
