import json
import os
import sys
from pathlib import Path

import pandas as pd
import joblib

sys.path.append(str(Path(__file__).parent))

from ra_pipeline import run_pipeline


def normalize_payload(payload):
    return {
        "Age": payload.get("age"),
        "Gender": payload.get("gender"),
        "Race": payload.get("race"),
        "BMI": payload.get("bmi"),
        "SmokingStatus": payload.get("smoking_status"),
        "Neutrophils": payload.get("neutrophils"),
        "Lymphocytes": payload.get("lymphocytes"),
        "CalorieConsumption": payload.get("dietary_calories"),
        "ProteinConsumption": payload.get("protein_consumption", 0.0),
        "CarbohydrateConsumption": payload.get("carb_consumption", 0.0),
        "FatConsumption": payload.get("fat_consumption", 0.0),
        "FiberConsumption": payload.get("fiber_consumption", 0.0),
        "CaffeineConsumption": payload.get("caffeine_consumption", 0.0),
        "EducationLevel": payload.get("education", "Unknown"),
        "MaritalStatus": payload.get("marital_status", "Unknown"),
        "FamilyIncome": payload.get("income", "Unknown"),
        "PhysicalActivity": payload.get("physical_activity", "Unknown"),
        "DrinkingStatus": payload.get("drinking_status", "Unknown"),
        "Hypertension": payload.get("hypertension", "Unknown"),
        "Diabetes": payload.get("diabetes", "Unknown"),
        "Hyperlipidemia": payload.get("hyperlipidemia", "Unknown"),
        "BRI": payload.get("bri", 0.0),
        "BRI_Trend": payload.get("bri_trend", 0.0),
        "DII": payload.get("dii", 0.0),
    }


def build_predictor():
    repo_root = Path(__file__).parent.parent
    default_data = repo_root / "data" / "The_final_data_after_screening.csv"
    raw_path = os.environ.get("RA_PIPELINE_DATA")

    if raw_path:
        data_path = Path(raw_path)
        if data_path.is_dir():
            data_path = data_path / "The_final_data_after_screening.csv"
    else:
        data_path = default_data

    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()

    try:
        data_path = data_path.relative_to(Path.cwd())
    except ValueError:
        data_path = Path(os.path.relpath(data_path, Path.cwd()))
    run_nested_cv = os.environ.get("RA_PIPELINE_NESTED_CV", "0") == "1"
    run_hypersearch = os.environ.get("RA_PIPELINE_HYPERSEARCH", "0") == "1"
    cache_path = os.environ.get(
        "RA_PIPELINE_CACHE",
        str(Path(__file__).parent / "model_cache.pkl"),
    )

    if cache_path and Path(cache_path).exists():
        try:
            return joblib.load(cache_path)
        except Exception:
            pass

    predictor, _, _ = run_pipeline(
        data_path=data_path,
        run_hypersearch=run_hypersearch,
        run_nested_cv=run_nested_cv,
    )

    if cache_path:
        try:
            joblib.dump(predictor, cache_path)
        except Exception:
            pass

    return predictor


def main():
    predictor = build_predictor()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            normalized = normalize_payload(payload)
            df = pd.DataFrame([normalized])
            result = predictor.predict(df).iloc[0]

            response = {
                "probability": float(result["RA_probability"]) / 100.0,
                "risk_level": str(result["RA_risk_level"]),
                "needs_followup": bool(result["needs_followup"]),
            }
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as exc:
            error = {"error": str(exc)}
            print(json.dumps(error))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
