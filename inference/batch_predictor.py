import pandas as pd
import joblib
import json
from pathlib import Path
from pipeline.feature_engineering import clean_data, build_rfm_features

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

class BatchPredictor:
    def __init__(self):
        self.scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
        self.kmeans = joblib.load(ARTIFACT_DIR / "kmeans.pkl")
        
        # Load schema to ensure feature order is identical to training
        with open(ARTIFACT_DIR / "feature_schema.json", "r") as f:
            self.schema = json.load(f)
            self.expected_features = self.schema["features"]

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw dataframe, cleans it, extracts features, and adds cluster labels.
        """
        # 1. Clean Data
        cleaned_df = clean_data(raw_df)

        # 2. Engineer Features
        customer_features = build_rfm_features(cleaned_df)

        # 3. Validate Columns
        X = customer_features[self.expected_features]

        # 4. Scale
        X_scaled = self.scaler.transform(X)

        # 5. Predict
        customer_features["Cluster"] = self.kmeans.predict(X_scaled)

        # Optional: Map cluster to business labels if you have them
        # customer_features["Segment_Label"] = customer_features["Cluster"].map({0: "Gold", 1: "Churn", ...})

        return customer_features