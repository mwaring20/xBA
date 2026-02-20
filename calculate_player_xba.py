"""
Calculates player xBA using three models from build_model.py and publishes a leaderboard
Also compares calculated xBA from models to expected
"""

import pandas as pd
from pathlib import Path

MODEL_DIR = Path("data/predictions")
INPUT_PATH = MODEL_DIR / "xba_predictions.parquet"
OUTPUT_PATH = MODEL_DIR / "player_xba_leaderboard.parquet"
CSV_OUTPUT_PATH = MODEL_DIR / "player_xba_leaderboard.csv"

MIN_AB = 50


# Calculates xBA for each player
def calculate_player_stats(df):
    grouped = (
        df.groupby(["batter", "batter_name"])
        .agg(
            at_bats=("is_at_bat", "sum"),
            hits=("is_hit", "sum"),
            sum_baseline=("baseline_xba", "sum"),
            sum_enhanced=("enhanced_xba", "sum"),
            sum_xgb=("xgb_xba", "sum"),
            sum_statcast=(
                "estimated_ba_using_speedangle",
                "sum"
            )
        )
        .reset_index()
    )

    # Actual BA
    grouped["actual_ba"] = (
        grouped["hits"]
        / grouped["at_bats"]
    )

    # Model xBAs
    grouped["baseline_xba"] = (
            grouped["sum_baseline"]
            / grouped["at_bats"]
    )

    grouped["enhanced_xba"] = (
            grouped["sum_enhanced"]
            / grouped["at_bats"]
    )

    grouped["xgb_xba"] = (
            grouped["sum_xgb"]
            / grouped["at_bats"]
    )

    grouped["statcast_xba"] = (
            grouped["sum_statcast"]
            / grouped["at_bats"]
    )

    # Comparisons
    grouped["baseline_minus_statcast"] = (
            grouped["baseline_xba"]
            - grouped["statcast_xba"]
    )

    grouped["baseline_minus_actual"] = (
            grouped["baseline_xba"]
            - grouped["actual_ba"]
    )

    grouped["enhanced_minus_statcast"] = (
            grouped["enhanced_xba"]
            - grouped["statcast_xba"]
    )

    grouped["enhanced_minus_actual"] = (
            grouped["enhanced_xba"]
            - grouped["actual_ba"]
    )

    grouped["xgb_minus_statcast"] = (
            grouped["xgb_xba"]
            - grouped["statcast_xba"]
    )

    grouped["xgb_minus_actual"] = (
            grouped["xgb_xba"]
            - grouped["actual_ba"]
    )

    return grouped


# Create leaderboard
def create_leaderboard(player_stats):
    leaderboard = player_stats.copy()
    # Apply minimum AB filter
    leaderboard = leaderboard[
        leaderboard["at_bats"] >= MIN_AB
    ]
    # Sort descending using xgb model
    leaderboard = leaderboard.sort_values(
        "xgb_xba",
        ascending=False
    )
    # Reset index
    leaderboard = leaderboard.reset_index(drop=True)
    return leaderboard


# Execution
if __name__ == "__main__":
    print("Loading predictions...")
    df = pd.read_parquet(INPUT_PATH)

    print("Calculating player stats...")
    player_stats = calculate_player_stats(df)

    print("Creating leaderboard...")
    leaderboard = create_leaderboard(player_stats)

    print("Saving leaderboard...")
    leaderboard.to_parquet(OUTPUT_PATH)
    leaderboard.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Saved parquet to {OUTPUT_PATH}")
    print(f"Saved csv to {CSV_OUTPUT_PATH}")
    print("\nTop 10 Players by xBA:\n")

    print(
        leaderboard[
            [
                "batter_name",
                "at_bats",
                "actual_ba",
                "statcast_xba",
                "xgb_xba",
                "xgb_minus_statcast"
            ]
        ].head(10)
    )

    print("Saving output...")
    player_stats.to_parquet(OUTPUT_PATH)

    print(f"Saved to {OUTPUT_PATH}")

    df = pd.read_parquet("data/predictions/xba_predictions.parquet")
