# Credit Card Fraud Detection (Imbalanced Dataset)

An end-to-end Machine Learning project focused on detecting fraudulent credit card transactions from highly imbalanced data using **SMOTE** (*Synthetic Minority Over-sampling Technique*) and **Precision-Recall (PR) Curve Optimization**.

---

## 📌 Project Overview

Credit card fraud detection is a critical application of machine learning in financial safety. However, fraud detection datasets present extreme class imbalance: legitimate transactions account for **~99.8%** of total records, while fraudulent transactions represent less than **0.2%**.

Standard evaluation metrics like accuracy fail in this domain. This project tackles class imbalance through synthetic oversampling and optimizes model thresholds specifically on Precision and Recall rather than default probability cutoffs.

---

## 🏗 Project Structure

```text
credit_card_fraud_detection/
├── data/
│   ├── raw/             # Raw transaction dataset (creditcard.csv)
│   └── processed/       # Processed & resampled datasets
├── src/
│   ├── __init__.py      # Package entry point
│   ├── data_loader.py   # Dataset loading logic
│   ├── preprocessing.py # Scaling, splits, and SMOTE resampling
│   ├── model.py         # Classifier definitions & training pipeline
│   └── evaluate.py      # Precision-Recall curve & threshold optimization
├── models/              # Saved model artifacts (.joblib / .pkl)
├── notebooks/           # Exploratory Data Analysis & visual experiments
├── .gitignore           # Git exclusions
├── requirements.txt     # Python dependencies
├── main.py              # Main execution pipeline
└── README.md            # Project documentation
```

---

## ⚙️ Key Methodologies

1. **Exploratory Data Analysis (EDA):** Class distribution inspection, transaction amount distribution, and time-feature analysis.
2. **Feature Preprocessing & Scaling:** Robust scaling for `Amount` and `Time` features.
3. **Synthetic Oversampling (SMOTE):** Handling class imbalance using SMOTE on the training set to prevent data leakage.
4. **Model Architecture:** Baseline Logistic Regression, Random Forest, and XGBoost classifiers.
5. **Precision-Recall Curve Optimization:** Tuning classification decision thresholds to maximize Recall while maintaining acceptable Precision.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

### 2. Set up virtual environment & dependencies
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📋 Project Roadmap

- [x] Step 0: Repository setup & directory structure
- [ ] Step 1: Data acquisition & Exploratory Data Analysis (EDA)
- [ ] Step 2: Preprocessing pipeline & SMOTE implementation
- [ ] Step 3: Model training (Logistic Regression, Random Forest, XGBoost)
- [ ] Step 4: Precision-Recall Curve evaluation & threshold tuning
- [ ] Step 5: Model serialization & deployment interface
