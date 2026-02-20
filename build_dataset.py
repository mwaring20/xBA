"""
Cleans dataset and engineers features to raw Statcast dataset
"""

import pandas as pd
import numpy as np
from pybaseball import statcast_sprint_speed, playerid_reverse_lookup
from pathlib import Path

DATA_DIR = Path("data/processed")  # Create path object to save processed output
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create directory and parent folder if missing


# Load raw statcast data
def load_raw_data(filepath):
    df = pd.read_parquet(filepath)  # Loads raw statcast parquet file into a pandas DataFrame
    return df


# Filter for official at bats
def filter_offical_at_bats(df):
    # Drop rows where events is None (i.e. not the final pitch of the at bat)
    df = df[df["events"].notna()].copy()

    # Events that DO NOT count as official at-bats
    non_ab_events = [
        "walk",
        "intent_walk",
        "hit_by_pitch",
        "sac_bunt",
        "sac_fly",
        "sac_fly_double_play",
        "catcher_interf",
        "truncated_pa",
    ]

    # Keep everything except non-AB events
    df = df[~df["events"].isin(non_ab_events)].copy()

    df["is_at_bat"] = 1

    df["is_hit"] = df["events"].isin([
        "single",
        "double",
        "triple",
        "home_run"
    ]).astype(int)

    return df


# Ensure only 1 row per at bat
def deduplicate_at_bats(df):

    df = df.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    )

    df = df.drop_duplicates(
        subset=["game_pk", "at_bat_number", "batter"],
        keep="last"
    )

    assert df.duplicated(
        subset=["game_pk", "at_bat_number", "batter"]
    ).sum() == 0

    return df


# Add batter name to dataset
def add_batter_names(df):
    unique_ids = df["batter"].dropna().unique()
    lookup = playerid_reverse_lookup(unique_ids, key_type="mlbam")
    lookup["batter_name"] = (
        lookup["name_first"] + " " + lookup["name_last"]
    )
    lookup = lookup.rename(
        columns={"key_mlbam": "batter"}
    )
    lookup = lookup[["batter", "batter_name"]]
    df = df.merge(
        lookup,
        on="batter",
        how="left"
    )
    return df


# Add spray angle
def add_spray_angle(df):
    """
    Convert hc_x and hc_y into a horizontal spray angle (degrees).
    125.42, 198.27 = Approximate home plate coordinates (Statcast reference frame)
    """
    df = df.copy()

    # Initialize spray_angle column with NaN
    df["spray_angle"] = np.nan

    # Mask where coordinates exist
    mask = df["hc_x"].notna() & df["hc_y"].notna()

    # Compute spray angle only for valid rows
    df.loc[mask, "spray_angle"] = np.degrees(
        np.arctan2(
            df.loc[mask, "hc_x"] - 125.42,
            198.27 - df.loc[mask, "hc_y"]
        )
    )

    return df


# Add nonlinear terms to the model
def add_nonlinear_terms(df):
    """
    Add squared and interaction terms for EV and LA (for logistic regression)
    EV and launch angle relationship to hit likelihood is not linear, adding curvature here
    Also capturing the interaction effect (95 mph at 10 degrees =/ 95 mph at 40 degrees)
    """
    df["launch_speed_sq"] = df["launch_speed"] ** 2
    df["launch_angle_sq"] = df["launch_angle"] ** 2
    df["ev_la_interaction"] = df["launch_speed"] * df["launch_angle"]
    return df


# Pull and merge sprint speed data
def merge_sprint_speed(df, season):
    """
    Merge sprint speed into pitch-level data.
    """
    sprint_df = statcast_sprint_speed(season)
    sprint_df = sprint_df.rename(columns={"player_id": "batter"})

    df = df.merge(
        sprint_df[["batter", "sprint_speed"]],
        on="batter",
        how="left"
    )

    # Fill missing sprint speeds with league median
    df["sprint_speed"] = df["sprint_speed"].fillna(
        df["sprint_speed"].median()
    )

    return df


# Select final modelling columns
def select_features(df):
    """
    Select modeling features and drop rows with missing critical values.
    """
    cols = [
        "batter",
        "batter_name",
        "player_name",
        "events",
        "launch_speed",
        "launch_angle",
        "launch_speed_sq",
        "launch_angle_sq",
        "ev_la_interaction",
        "spray_angle",
        "sprint_speed",
        "is_at_bat",
        "is_hit",
        "estimated_ba_using_speedangle"
    ]

    return df[cols].copy()


# Save processed dataset
def save_dataset(df, filepath):
    """
    Save processed modeling dataset to parquet.
    """
    df.to_parquet(filepath, index=False)


# Pipeline execution
if __name__ == "__main__":

    RAW_PATH = Path("data/raw/statcast_2025_raw.parquet")
    OUTPUT_PATH = Path("data/processed/statcast_2025_modeling.parquet")
    SEASON = 2025

    print("Loading raw data...")
    df = load_raw_data(RAW_PATH)

    print("Filtering balls in play and strikeouts...")
    df = filter_offical_at_bats(df)

    print("Deduplicating at bats...")
    df = deduplicate_at_bats(df)

    print("Adding batter names...")
    df = add_batter_names(df)

    print("Adding spray angle...")
    df = add_spray_angle(df)

    print("Adding nonlinear terms...")
    df = add_nonlinear_terms(df)

    print("Merging sprint speed...")
    df = merge_sprint_speed(df, SEASON)

    print("Selecting final features...")
    df = select_features(df)

    print("Saving processed dataset...")
    save_dataset(df, OUTPUT_PATH)

    print("Dataset build complete.")
