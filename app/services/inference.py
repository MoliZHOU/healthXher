import numpy as np
import joblib
import os
import sys
import pandas as pd
from pathlib import Path
from app.core.config import settings

# Add project root to sys.path to allow importing from python.ra_pipeline
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from python.ra_pipeline import run_pipeline, RAPredictor
except ImportError:
    # Fallback if structure is different
    from python.ra_pipeline import run_pipeline, RAPredictor

class CriticalStartupError(Exception):
    """Raised when the inference engine cannot be initialized."""
    pass

class RAPredictorService:
    def __init__(self):
        self.predictor: RAPredictor = None
        self._load_or_train()

    def _load_or_train(self):
        cache_path = settings.PIPELINE_CACHE_PATH
        
        # 1. Try to load from cache
        if os.path.exists(cache_path):
            try:
                self.predictor = joblib.load(cache_path)
                print(f"[Inference] Loaded cached RAPredictor from {cache_path}")
                return
            except Exception as e:
                print(f"[Warning] Failed to load cache: {e}. Re-training...")

        # 2. Train if cache missing or failed
        print(f"[Inference] Initializing RA Research Engine using: {settings.TRAINING_DATA_PATH}")
        try:
            # Dynamically pass custom configurations to the pipeline
            predictor, _, _ = run_pipeline(
                data_path=settings.TRAINING_DATA_PATH,
                custom_weights=settings.CUSTOM_FEATURE_WEIGHTS, # New scalable param
                run_hypersearch=False,
                run_nested_cv=False
            )
            self.predictor = predictor
            
            # 3. Save to cache
            joblib.dump(self.predictor, cache_path)
            print(f"[Inference] Training complete. RAPredictor cached at {cache_path}")
        except Exception as e:
            raise CriticalStartupError(f"Failed to initialize RAPredictor: {e}")

    def predict(self, features_dict: dict) -> float:
        if self.predictor is None:
            raise CriticalStartupError("Predictor not initialized.")

        # The RAPredictor expects a DataFrame
        df_input = pd.DataFrame([features_dict])
        
        try:
            # The pipeline handles derived features (NLR, BRI, DII) automatically
            results = self.predictor.predict(df_input)
            # The pipeline adds 'RA_probability' (0-100)
            probability = float(results['RA_probability'].iloc[0]) / 100.0
            return probability
        except Exception as e:
            raise CriticalStartupError(f"Inference failed: {e}")

# Singleton instance
predictor_service = RAPredictorService()
