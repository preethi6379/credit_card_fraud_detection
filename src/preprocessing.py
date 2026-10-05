"""
Preprocessing pipeline including feature scaling, train-test splitting,
and SMOTE oversampling for imbalanced data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


def preprocess_data(df: pd.DataFrame, target_col: str = "Class", test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target, scales 'Time' and 'Amount', and performs train-test split.
    
    Note: SMOTE is ONLY applied to the training set to prevent data leakage!
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Robust scaling for skewed features Time and Amount if present
    scaler = RobustScaler()
    for col in ["Time", "Amount"]:
        if col in X.columns:
            X[col] = scaler.fit_transform(X[[col]])

    # Train-Test Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Applies SMOTE oversampling on training data only.
    """
    print("[+] Applying SMOTE on training set...")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[+] Resampled training shape: {X_train_res.shape} | Positive class count: {y_train_res.sum()}")
    return X_train_res, y_train_res
