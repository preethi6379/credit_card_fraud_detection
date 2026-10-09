import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import load_data
def prepare_data(test_size:float =0.2,random_state:int = 42):
    df = load_data()
    x = df.drop(columns=["Class"])
    y = df["Class"]
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=random_state,stratify=y)
    print(f"[+] Train set: {X_train.shape[0]:,} rows (frauds: {y_train.sum()})")
    print(f"[+] Test set : {X_test.shape[0]:,} rows (Frauds: {y_test.sum()})")
    
    return X_train, X_test, y_train, y_test
def train_logistic_regression(X_train, y_train, random_state: int = 42):
    print("\n[+] Training Logistic Regression with SMOTE Pipeline...")
    pipeline = Pipeline([
        ('scalar',RobustScaler()),
        ('smote',SMOTE(random_state = random_state)),
        ('classifier',LogisticRegression(max_iter =1000,random_state=random_state))
    ])
    pipeline.fit(X_train,y_train)
    print("[+] Logistic Regression Training Complete!")
    os.makedirs("models",exist_ok=True)
    model_path = "models/logistic_regression_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[+] Saved model to: {model_path}")
    return pipeline
def train_random_forest(X_train,y_train,n_estimators: int = 100,random_state: int =42):
    print("\n[+] Training Random Forest with SMOTE Pipeline...")
    pipeline = Pipeline([
        ('scaler',RobustScaler()),
        ('smote',SMOTE(random_state=random_state)),
        ('classifier',RandomForestClassifier(n_estimators=n_estimators,random_state=random_state,n_jobs=-1))
    ])
    pipeline.fit(X_train,y_train)
    print("[+] Random Forest Training Compelete!")
    os.makedirs("models",exist_ok=True)
    model_path = "models/random_forest_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[+] Saved model to: {model_path}")
    return pipeline

if __name__ == "__main__":
    X_train,X_test,y_train,y_test = prepare_data()
    lr_pipeline = train_logistic_regression(X_train,y_train)
    rf_pipeline = train_random_forest(X_train,y_train)
    print("\n[+] ALL MODELS TRAINED AND SAVED SUCCESSFULLY IN models/ FOLDER!")