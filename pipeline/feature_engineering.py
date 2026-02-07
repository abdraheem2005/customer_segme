import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs standard data cleaning on the raw transaction set.
    """
    # Create copy to avoid SettingWithCopy warnings on the original slice
    df = df.copy()

    # 1. Drop rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # 2. Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 3. Handle negative quantities (returns) if necessary,
    # OR remove cancelled transactions (InvoiceNo starts with 'C')
    # Based on your PDF, we strictly remove cancelled invoices:
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # 4. Remove invalid prices/quantities
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 5. Calculate TotalPrice for the Monetary feature
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df

def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data into Customer-Level RFM + Behavioral features.
    """
    # Define a snapshot date (usually max date + 1 day to ensure Recency > 0)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # --- RFM Aggregation ---
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days, # Recency
        "InvoiceNo": "nunique",                                  # Frequency
        "TotalPrice": "sum"                                      # Monetary
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # --- Behavioral Aggregation ---
    behavior = df.groupby("CustomerID").agg({
        "Quantity": "sum",       # TotalQuantity
        "StockCode": "nunique"   # UniqueProducts
    }).reset_index()

    behavior.columns = ["CustomerID", "TotalQuantity", "UniqueProducts"]

    # --- Merge ---
    final_features = rfm.merge(behavior, on="CustomerID", how="left")

    return final_features