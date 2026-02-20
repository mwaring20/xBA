"""
Pulls Statcast dataset for 2025 season
"""

import pybaseball
from pathlib import Path

DATA_DIR = Path("data/raw")  # Create path object to save raw output
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create directory and parent folder if missing


def pull_statcast_2025():
    print("Pulling Statcast data...")
    # Full 2025 season including post season
    df = pybaseball.statcast("2025-03-27", "2025-11-30")

    output_path = DATA_DIR / "statcast_2025_raw.parquet"  # Output file path
    df.to_parquet(output_path)  # Save output file

    print(f"Saved raw data to {output_path}")  # Save confirmation
    print(f"Rows: {len(df):,}")  # Number of rows in output file


if __name__ == "__main__":  # Statcast pull not run unless this file is run directly
    pull_statcast_2025()
