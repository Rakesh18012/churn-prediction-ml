# 🤖 ML Churn Prediction Pipeline & Feature Store

> End-to-end ML pipeline for customer churn prediction: feature engineering from raw transactional data, **XGBoost model (AUC: 0.91)**, experiment tracking with MLflow, and a versioned **feature store on AWS S3** — with zero-downtime retraining.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green) ![MLflow](https://img.shields.io/badge/MLflow-2.8-orange) ![AWS](https://img.shields.io/badge/AWS-S3-yellow) ![Scikit](https://img.shields.io/badge/Scikit--learn-1.3-blue)

---

## 🏗️ Architecture

```
Raw Transactional Data (S3)
        │
        ▼
  Feature Engineering Pipeline
  ├── Recency / Frequency / Monetary (RFM)
  ├── Behavioral features (session length, page views)
  ├── Temporal features (day of week, days since last order)
  └── Engineered ratios and rolling stats
        │
        ▼
  Feature Store (AWS S3, versioned)
  └── features/v{n}/train.parquet, test.parquet
        │
        ▼
  Model Training
  ├── XGBoost Classifier
  ├── Hyperparameter tuning (Optuna)
  ├── Cross-validation (StratifiedKFold, k=5)
  └── MLflow experiment tracking
        │
        ▼
  Model Registry (MLflow)
  └── Staging → Production promotion
        │
        ▼
  Batch Prediction & Scoring API
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.91** |
| Precision | 0.87 |
| Recall | 0.83 |
| F1 Score | 0.85 |
| Accuracy | 0.89 |

---

## 📁 Project Structure

```
churn-prediction-ml/
├── src/
│   ├── features/
│   │   ├── feature_engineering.py    # Feature computation
│   │   └── feature_store.py          # S3 feature store read/write
│   ├── training/
│   │   ├── train.py                  # Main training script
│   │   ├── hyperparameter_tuning.py  # Optuna tuning
│   │   └── evaluate.py               # Model evaluation
│   └── serving/
│       └── predict.py                # Batch scoring
├── notebooks/
│   └── EDA.ipynb                     # Exploratory analysis
├── tests/
│   └── test_features.py
├── mlflow/
│   └── MLproject
├── requirements.txt
└── README.md
```

---

## 🔧 Quick Start

```bash
git clone https://github.com/Rakesh18012/churn-prediction-ml
cd churn-prediction-ml
pip install -r requirements.txt

# Start MLflow UI
mlflow ui --port 5000 &

# Run feature engineering
python src/features/feature_engineering.py --input s3://your-bucket/raw/

# Train model
python src/training/train.py --experiment-name churn_v1

# View results at http://localhost:5000
```
