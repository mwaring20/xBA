"""
Builds and trains three xBA models:
1. Recreate Statcast xBA
2. Enhance Statcast xBA calculation by adding additional features
3. Build Gradient Boost model to calculate xBA
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss
)
from xgboost import XGBClassifier

MODEL_DIR = Path("data/models")  # Create path object to save csv output
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Create directory and parent folder if missing


# Load Dataset created in build_dataset.py
def load_dataset(filepath):
    df = pd.read_parquet(filepath)
    return df


# Train/Test Split
def split_data(df):
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )
    return train, test


# Drop rows with missing values (only model xBA using balls in play)
def prepare_model_data(df, feature_cols, target_col):
    model_df = df.dropna(subset=feature_cols + [target_col]).copy()
    X = model_df[feature_cols]
    y = model_df[target_col]
    return X, y, model_df


# Define Feature Sets
def get_feature_sets():
    baseline_features = [  # Baseline features to recreate statcast xBA
        "launch_speed",
        "launch_angle",
        "launch_speed_sq",
        "launch_angle_sq",
        "ev_la_interaction"
    ]
    enhanced_features = baseline_features + [  # Include additional features to improve xBA
        "spray_angle",
        "sprint_speed"
    ]
    target = "is_hit"
    return baseline_features, enhanced_features, target


# Train Logistic Regression
def train_logistic(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model


# Train XGBoost Model
def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# Evaluate Model
def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]  # Predicts probability a batted ball becomes a hit
    auc = roc_auc_score(y_test, probs)  # Measure ranking performance
    ll = log_loss(y_test, probs)  # Measures probability accuracy
    brier = brier_score_loss(y_test, probs)  # Measure calibration accuracy
    return {
        "AUC": auc,
        "LogLoss": ll,
        "Brier": brier
    }


# Correlation with Statcast xBA
def evaluate_xba_correlation(model, X_test, statcast_xba):  # How closely does baseline match statcast
    probs = model.predict_proba(X_test)[:, 1]
    corr = np.corrcoef(probs, statcast_xba)[0, 1]
    return corr


# Save Predictions
def save_predictions(test_df, baseline_model, enhanced_model, xgb_model,
                     baseline_features, enhanced_features):

    output = test_df.copy()

    # Baseline predictions (only where features exist)
    mask_base = output[baseline_features].notna().all(axis=1)
    output.loc[mask_base, "baseline_xba"] = baseline_model.predict_proba(
        output.loc[mask_base, baseline_features]
    )[:, 1]

    # Enhanced predictions
    mask_enh = output[enhanced_features].notna().all(axis=1)
    output.loc[mask_enh, "enhanced_xba"] = enhanced_model.predict_proba(
        output.loc[mask_enh, enhanced_features]
    )[:, 1]

    # XGBoost predictions
    output.loc[mask_enh, "xgb_xba"] = xgb_model.predict_proba(
        output.loc[mask_enh, enhanced_features]
    )[:, 1]

    output_path = MODEL_DIR / "xba_predictions.parquet"
    output.to_parquet(output_path)

    print(f"\nSaved predictions to {output_path}")


# Save Metrics
def save_metrics(baseline_metrics, enhanced_metrics, xgb_metrics, baseline_corr):
    results = pd.DataFrame({
        "Model": [
            "Baseline Logistic",
            "Enhanced Logistic",
            "XGBoost"
        ],
        "AUC": [
            baseline_metrics["AUC"],
            enhanced_metrics["AUC"],
            xgb_metrics["AUC"]
        ],
        "LogLoss": [
            baseline_metrics["LogLoss"],
            enhanced_metrics["LogLoss"],
            xgb_metrics["LogLoss"]
        ],
        "Brier": [
            baseline_metrics["Brier"],
            enhanced_metrics["Brier"],
            xgb_metrics["Brier"]
        ],
        "Correlation_with_Statcast": [
            baseline_corr,
            None,
            None
        ]
    })
    output_path = MODEL_DIR / "model_comparison.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved model comparison to {output_path}")


# Save Models
def save_models(baseline_model, enhanced_model, xgb_model):
    joblib.dump(baseline_model,
                MODEL_DIR / "baseline_logistic.pkl")
    joblib.dump(enhanced_model,
                MODEL_DIR / "enhanced_logistic.pkl")
    joblib.dump(xgb_model,
                MODEL_DIR / "xgboost.pkl")
    print("Saved trained models")


# Main Execution
if __name__ == "__main__":
    DATA_PATH = Path(
        "data/processed/statcast_2025_modeling.parquet"
    )

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("Splitting data...")
    train, test = split_data(df)
    baseline_features, enhanced_features, target = get_feature_sets()

    # Model 1: Baseline Logistic
    print("\nTraining Baseline Logistic Model...")

    X_train_base, y_train_base, train_base_df = prepare_model_data(
        train, baseline_features, target
    )
    X_test_base, y_test_base, test_base_df = prepare_model_data(
        test, baseline_features, target
    )

    baseline_model = train_logistic(X_train_base, y_train_base)

    baseline_metrics = evaluate_model(
        baseline_model,
        X_test_base,
        y_test_base
    )

    baseline_corr = evaluate_xba_correlation(
        baseline_model,
        X_test_base,
        test_base_df["estimated_ba_using_speedangle"]
    )

    # Model 2: Enhanced Logistic
    print("\nTraining Enhanced Logistic Model...")

    X_train_enh, y_train_enh, train_enh_df = prepare_model_data(
        train, enhanced_features, target
    )
    X_test_enh, y_test_enh, test_enh_df = prepare_model_data(
        test, enhanced_features, target
    )

    enhanced_model = train_logistic(X_train_enh, y_train_enh)

    enhanced_metrics = evaluate_model(
        enhanced_model,
        X_test_enh,
        y_test_enh
    )

    # Model 3: XGBoost
    print("\nTraining XGBoost Model...")

    xgb_model = train_xgboost(X_train_enh, y_train_enh)

    xgb_metrics = evaluate_model(
        xgb_model,
        X_test_enh,
        y_test_enh
    )

    # Print Results
    print("\n==============================")

    print("MODEL COMPARISON")

    print("==============================")

    print("\nBaseline Logistic:")

    print(baseline_metrics)

    print("Correlation with Statcast xBA:", baseline_corr)

    print("\nEnhanced Logistic:")

    print(enhanced_metrics)

    print("\nXGBoost:")

    print(xgb_metrics)

    save_models(
        baseline_model,
        enhanced_model,
        xgb_model
    )

    save_predictions(
        test,
        baseline_model,
        enhanced_model,
        xgb_model,
        baseline_features,
        enhanced_features
    )

    save_metrics(
        baseline_metrics,
        enhanced_metrics,
        xgb_metrics,
        baseline_corr
    )

    print("\nDone.")

