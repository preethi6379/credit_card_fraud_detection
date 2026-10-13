import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import prepare_data
def load_trained_models():
    """
    Loads saved pipeline models from the models/ directory.
    """
    lr_path = "models/logistic_regression_pipeline.joblib"
    rf_path = "models/random_forest_pipeline.joblib"

    if not os.path.exists(lr_path) or not os.path.exists(rf_path):
        raise FileNotFoundError("Trained models not found in models/ folder. Please run src/train.py first!")

    print("[+] Loading trained models from models/ directory...")
    lr_pipeline = joblib.load(lr_path)
    rf_pipeline = joblib.load(rf_path)
    print("[+] Successfully loaded Logistic Regression and Random Forest models!")

    return lr_pipeline, rf_pipeline

def evaluate_models(lr_pipeline,rf_pipeline,X_test,y_test):
    lr_probs = lr_pipeline.predict_proba(X_test)[:,1]
    rf_probs = rf_pipeline.predict_proba(X_test)[:,1]
    lr_pr_auc = average_precision_score(y_test,lr_probs)
    rf_pr_auc = average_precision_score(y_test,rf_probs)
    print(f"\n--- PR-AUC Comparison ---")
    print(f"Logistic Regression PR-AUC : {lr_pr_auc:.4f}")
    print(f"Random Forest PR-AUC       : {rf_pr_auc:.4f}")
    lr_precision, lr_recall, lr_thresholds = precision_recall_curve(y_test, lr_probs)
    rf_precision, rf_recall, rf_thresholds = precision_recall_curve(y_test, rf_probs)
        # Calculate F1 scores across all thresholds
    lr_f1 = np.where((lr_precision + lr_recall) == 0, 0, 2 * (lr_precision * lr_recall) / (lr_precision + lr_recall))
    rf_f1 = np.where((rf_precision + rf_recall) == 0, 0, 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall))
        # Find index of the highest F1-score
    best_lr_idx = np.argmax(lr_f1)
    best_rf_idx = np.argmax(rf_f1)
    
    # Get the corresponding optimal threshold
    best_lr_threshold = lr_thresholds[best_lr_idx]
    best_rf_threshold = rf_thresholds[best_rf_idx]
    print(f"\n--- Optimal Thresholds & Best F1-Scores ---")
    print(f"Logistic Regression : Threshold = {best_lr_threshold:.4f} | Best F1 = {lr_f1[best_lr_idx]:.4f}")
    print(f"Random Forest       : Threshold = {best_rf_threshold:.4f} | Best F1 = {rf_f1[best_rf_idx]:.4f}")
    
    return lr_precision, lr_recall, rf_precision, rf_recall, lr_pr_auc, rf_pr_auc
def plot_precision_recall_curves(lr_precision, lr_recall, rf_precision, rf_recall, lr_pr_auc, rf_pr_auc):
    plt.figure(figsize=(10,6))
    plt.plot(rf_recall,rf_precision,label=f"Random Forest (PR-AUC = {rf_pr_auc:.3f})",color="green",linewidth=2)
    plt.plot(lr_recall,lr_precision,label =f"Logistic Regression (PR-AUC ={lr_pr_auc:.3f})",color="blue",linestyle="--")
        # 3. Add labels and styling
    plt.xlabel("Recall (Fraction of Frauds Caught)", fontsize=12)
    plt.ylabel("Precision (Accuracy of Flagged Frauds)", fontsize=12)
    plt.title("Precision-Recall Curve Comparison (SMOTE Resampled)", fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 4. Save image
    plot_path = "models/precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved Precision-Recall curve graph to: {plot_path}")
    plt.close()
if __name__ == "__main__":
    # 1. Prepare unseen test data
    _, X_test, _, y_test = prepare_data()
    
    # 2. Load trained models from models/
    lr_pipeline, rf_pipeline = load_trained_models()
    
    # 3. Evaluate & get curve metrics
    lr_p, lr_r, rf_p, rf_r, lr_auc, rf_auc = evaluate_models(lr_pipeline, rf_pipeline, X_test, y_test)
    
    # 4. Generate and save graph
    plot_precision_recall_curves(lr_p, lr_r, rf_p, rf_r, lr_auc, rf_auc)