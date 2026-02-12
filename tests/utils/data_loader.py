import pandas as pd
import random
import os
import math

class SmartDataLoader:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)

    def get_random_patient(self) -> dict:
        row = self.df.sample(n=1).iloc[0]
        
        # Prioritize Raw NHANES columns if they exist
        # Screened CSV mapping:
        payload = {
            "age": int(row.get("Age", 40)),
            "gender": str(row.get("Gender", "Female")),
            "smoking_status": str(row.get("SmokingStatus", "Never")),
            "fiber_consumption": float(row.get("FiberConsumption", 15.0)),
            "bmi": float(row.get("BMI", 25.0)),
            "physical_activity": str(row.get("PhysicalActivity", "Sedentary")),
            "hypertension": str(row.get("Hypertension", "Normal")),
            "diabetes": str(row.get("Diabetes", "Normal")),
        }

        # Check for raw measurement columns first (Refinement 1)
        # Note: headers in primary CSV were BMXWAIST, LBDNENO, etc.
        # headers in screened CSV were BRI, NLR (sometimes)
        
        waist = row.get("BMXWAIST")
        height = row.get("BMXHT") # Height might be missing in some cycles
        
        if pd.notnull(waist):
            payload["waist_cm"] = float(waist)
        
        if pd.notnull(height):
            payload["height_cm"] = float(height)
        
        # Fallback for BRI reverse-mapping if raw columns missing but BRI exists
        if "waist_cm" not in payload and "BRI" in row:
            # Reverse map: assume height 170cm
            h = 170.0
            bri = float(row["BRI"])
            # Derived from BRI formula: e^2 = (1 - (BRI-364.2)/-365.5)^2
            # (waist/(2pi) / (0.5h))^2 = 1 - e^2
            try:
                e_val = (bri - 364.2) / -365.5
                inner = 1 - (e_val ** 2)
                if inner > 0:
                    waist_val = math.sqrt(inner) * (0.5 * h) * (2 * 3.14159)
                    payload["waist_cm"] = round(waist_val, 2)
                    payload["height_cm"] = h
            except:
                payload["waist_cm"] = 90.0
                payload["height_cm"] = 170.0

        # NLR / Neutrophils
        neutro = row.get("LBDNENO")
        lympho = row.get("LBDLYMNO")
        if pd.notnull(neutro) and pd.notnull(lympho):
            payload["neutrophils"] = float(neutro)
            payload["lymphocytes"] = float(lympho)
        elif "NLR" in row:
            # Mock split 2:1
            nlr = float(row["NLR"])
            payload["neutrophils"] = nlr * 1.5
            payload["lymphocytes"] = 1.5

        # DII
        if "DII" in row:
            payload["dii"] = float(row["DII"])

        return payload, row.get("SEQN", "UNKNOWN")
