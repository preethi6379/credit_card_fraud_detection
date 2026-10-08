import os
import pandas as pd

def load_data(file_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Loads raw CSV data from file_path and removes duplicates.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at '{file_path}'. Please check file path!")
    
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    print(f"[+] Loaded {len(df):,} clean transaction rows.")
    return df