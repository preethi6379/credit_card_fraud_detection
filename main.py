import os
from src.train import prepare_data,train_logistic_regression,train_random_forest
from src.predict import load_trained_models,evaluate_models,plot_precision_recall_curves
def main():
    print("==================================================")
    print("  Credit Card Fraud Detection ML Pipeline         ")
    print("==================================================")
    
    # Step 1: Prepare train and test datasets
    X_train, X_test, y_train, y_test = prepare_data()
        # Step 2: Check if models exist on disk; if not, train them
    lr_path = "models/logistic_regression_pipeline.joblib"
    rf_path = "models/random_forest_pipeline.joblib"
    
    if not os.path.exists(lr_path) or not os.path.exists(rf_path):
        print("\n[+] Models not found on disk. Training models now...")
        train_logistic_regression(X_train, y_train)
        train_random_forest(X_train, y_train)
    else:
        print("\n[+] Found existing trained models in models/ folder! Skipping retraining.")
    
    # Step 3: Load the trained pipelines
    lr_pipeline, rf_pipeline = load_trained_models()

       # Step 4: Evaluate models & compute optimal thresholds
    lr_p, lr_r, rf_p, rf_r, lr_auc, rf_auc = evaluate_models(lr_pipeline, rf_pipeline, X_test, y_test)
    
    # Step 5: Generate and save Precision-Recall curve graph
    plot_precision_recall_curves(lr_p, lr_r, rf_p, rf_r, lr_auc, rf_auc)
    
    print("\n==================================================")
    print("  Pipeline Completed Successfully!                ")
    print("==================================================")


if __name__ == "__main__":
    main()