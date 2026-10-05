"""
Data loading utilities for Credit Card Fraud Detection dataset.
"""

import os
import pandas as pd

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw credit card dataset from a CSV file.

    Parameters:
        file_path (str): Path to the creditcard.csv file.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please download creditcard.csv into data/raw/.")
    
    print(f"[+] Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"[+] Dataset shape: {df.shape}")
    return df
