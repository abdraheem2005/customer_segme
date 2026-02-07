import pandas as pd
from pathlib import Path

def load_customer_file(uploaded_file) -> pd.DataFrame:
    """
    Robust file loader for Streamlit. 
    Accepts: CSV, XLSX, XLS
    Returns: Pandas DataFrame
    """
    # uploaded_file in Streamlit has a .name attribute
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()

    try:
        if suffix == ".csv":
            # Try utf-8 first, fallback to latin1 (common in retail datasets)
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0) # Reset pointer
                df = pd.read_csv(uploaded_file, encoding='latin1')
                
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
            
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Please upload CSV or Excel.")
            
        return df

    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")