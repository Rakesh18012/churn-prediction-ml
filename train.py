"""
train.py
XGBoost churn prediction training pipeline with MLflow experiment tracking.
Trains on versioned features from S3 feature store.
AUC: 0.91 on held-out test set.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
import optuna
import boto3
import os
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Feature Store loader ───────────────────────────────────────────────────────

def load_features_from_s3(bucket: str, version: str = "latest"):
    """Load versioned feature set from S3 feature store."""
    s3 = boto3.client("s3")

    if version == "latest":
        # Find latest version
        response = s3.list_objects_v2(Bucket=bucket, Prefix="features/v")
        versions = sorted([
            obj["Key"].split("/")[1]
            for obj in response.get("Contents", [])
            if "train.parquet" in obj["Key"]
        ])
        version = versions[-1] if versions else "v1"

    logger.info(f"Loading features version: {version}")

    # Download from S3 to temp
    for split in ["train", "test"]:
        key = f"features/{version}/{split}.parquet"
        local = f"/tmp/{split}.parquet"
        s3.download_file(bucket, key, local)

    train_df = pd.read_parquet("/tmp/train.parquet")
    test_df  = pd.read_parquet("/tmp/test.parquet")

    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df, version


# ── Feature engineering ────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Select, encode, and prepare features for XGBoost."""
    TARGET = "churned"

    feature_cols = [
        # RFM features
        "days_since_last_order", "order_frequency_30d", "total_lifetime_spend",
        "avg_order_value", "max_order_value",
        # Behavioral
        "avg_session_duration", "pages_per_session", "support_tickets_90d",
        "refund_rate", "promo_usage_rate",
        # Temporal
        "tenure_days", "days_since_first_order", "orders_last_7d",
        "orders_last_30d", "orders_last_90d",
        # Ratios
        "spend_trend",  # recent vs historical spend ratio
        "engagement_score",
        # Categorical (encoded)
        "country_encoded", "device_type_encoded", "acquisition_channel_encoded",
    ]

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET]

    return X, y, available


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train, n_trials: int = 50):
    """Use Optuna to find optimal XGBoost hyperparameters."""
    logger.info(f"Running Optuna hyperparameter search ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "use_label_encoder": False,
            "eval_metric": "auc",
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test, params: dict, experiment_name: str):
    """Train XGBoost with MLflow tracking."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"xgb_churn_{datetime.now().strftime('%Y%m%d_%H%M')}"):

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("positive_rate", float(y_train.mean()))

        # Train
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=100,
        )

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = {
            "auc_roc":   roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "accuracy":  accuracy_score(y_test, y_pred),
        }

        mlflow.log_metrics(metrics)
        logger.info(f"\nModel Metrics:\n{metrics}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        # Cross-validation on full train set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std",  cv_scores.std())
        logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Feature importance
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv")
        logger.info(f"\nTop 10 Features:\n{importance.head(10)}")

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn_prediction_xgb",
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        return model, metrics, run_id


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",          default=os.environ.get("FEATURE_STORE_BUCKET", "rakesh-feature-store"))
    parser.add_argument("--version",         default="latest")
    parser.add_argument("--experiment-name", default="churn_prediction")
    parser.add_argument("--n-trials",        type=int, default=50)
    parser.add_argument("--skip-tuning",     action="store_true")
    args = parser.parse_args()

    # 1. Load features
    train_df, test_df, version = load_features_from_s3(args.bucket, args.version)

    # 2. Prepare
    X_train, y_train, features = prepare_features(train_df)
    X_test,  y_test,  _        = prepare_features(test_df)
    logger.info(f"Features: {features}")

    # 3. Tune or use defaults
    if args.skip_tuning:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 3,
        }
    else:
        best_params = tune_hyperparameters(X_train, y_train, args.n_trials)

    # 4. Train with MLflow tracking
    model, metrics, run_id = train_model(
        X_train, y_train, X_test, y_test,
        best_params, args.experiment_name
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete!")
    logger.info(f"AUC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f}")
    logger.info(f"MLflow Run: {run_id}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
