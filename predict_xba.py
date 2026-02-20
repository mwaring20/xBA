"""
Predicts xBA using models built in build_model.py
"""

import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path("data/models")
INPUT_PATH = Path(
    "data/processed/statcast_2025_modeling.parquet"
)
OUTPUT_DIR = Path("data/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "xba_predictions.parquet"


# Load trained models from build_model.py (Generate predictions without retraining models)
def load_models():
    baseline = joblib.load(
        MODEL_DIR / "baseline_logistic.pkl"  # Baseline logistic model
    )
    enhanced = joblib.load(
        MODEL_DIR / "enhanced_logistic.pkl"  # Enhanced logistic model
    )
    xgb = joblib.load(
        MODEL_DIR / "xgboost.pkl"  # XGBoost model
    )
    return baseline, enhanced, xgb


#
def get_feature_sets():
    # Recreate Statcast xBA
    baseline_features = [
        "launch_speed",
        "launch_angle",
        "launch_speed_sq",
        "launch_angle_sq",
        "ev_la_interaction"
    ]
    # Add sprint speed and spray angle
    enhanced_features = baseline_features + [
        "spray_angle",
        "sprint_speed"
    ]
    return baseline_features, enhanced_features


# Apply models
def generate_predictions(df, baseline, enhanced, xgb):
    baseline_features, enhanced_features = get_feature_sets()

    df = df.copy()

    # Initialize prediction columns as NaN
    df["baseline_xba"] = float("nan")
    df["enhanced_xba"] = float("nan")
    df["xgb_xba"] = float("nan")

    # Baseline predictions
    mask_base = df[baseline_features].notna().all(axis=1)

    if mask_base.any():
        df.loc[mask_base, "baseline_xba"] = baseline.predict_proba(
            df.loc[mask_base, baseline_features]
        )[:, 1]

    # Enhanced predictions
    mask_enh = df[enhanced_features].notna().all(axis=1)

    if mask_enh.any():
        df.loc[mask_enh, "enhanced_xba"] = enhanced.predict_proba(
            df.loc[mask_enh, enhanced_features]
        )[:, 1]

        df.loc[mask_enh, "xgb_xba"] = xgb.predict_proba(
            df.loc[mask_enh, enhanced_features]
        )[:, 1]

    # Set non-batted-ball events to 0 (e.g. strikeouts where launch_speed is NaN)
    non_bip_mask = df["launch_speed"].isna()
    df.loc[non_bip_mask, ["baseline_xba",
                          "enhanced_xba",
                          "xgb_xba"]] = 0.0

    return df


# Execution
if __name__ == "__main__":
    print("Loading models...")

    baseline, enhanced, xgb = load_models()

    print("Loading data...")

    df = pd.read_parquet(INPUT_PATH)

    print("Generating predictions...")

    df = generate_predictions(
        df,
        baseline,
        enhanced,
        xgb
    )

    print("Saving predictions...")

    df.to_parquet(OUTPUT_PATH)

    print(f"Saved to {OUTPUT_PATH}")

