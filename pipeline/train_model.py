import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from feature_engineering import clean_data, build_rfm_features

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "data" / "customer_segmentation.csv"  # Put your training data here

def train():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    print(f"   Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Please place your training CSV at {DATA_PATH}")
    
    raw_df = pd.read_csv(DATA_PATH, encoding='latin1')
    
    # 2. Clean & Engineer Features (Using the logic we built earlier)
    print("   Cleaning and Engineering Features...")
    df_clean = clean_data(raw_df)
    features_df = build_rfm_features(df_clean)
    
    # Define features exactly as in your PDF
    feature_cols = [
        "Recency", 
        "Frequency", 
        "Monetary", 
        "TotalQuantity", 
        "UniqueProducts"
    ]
    
    X = features_df[feature_cols]
    
    # 3. Scale
    print("   Scaling Data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train KMeans (k=4 based on your PDF's Elbow method)
    print("   Training KMeans (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 5. Save Artifacts
    print("   Saving Artifacts...")
    ARTIFACT_DIR.mkdir(exist_ok=True)
    
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
    joblib.dump(kmeans, ARTIFACT_DIR / "kmeans.pkl")
    
    # Save schema for validation later
    schema = {"features": feature_cols}
    with open(ARTIFACT_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f)
        
    print("✅ Training Complete. Artifacts saved to /artifacts")

if __name__ == "__main__":
    train()