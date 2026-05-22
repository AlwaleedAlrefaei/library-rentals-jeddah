"""Data loading and cleaning for the Jeddah library rentals dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


NUMERIC_COLS = [
    "Temperature_C",
    "Humidity_pct",
    "Wind_Speed_ms",
    "Visibility_m",
    "Solar_Radiation_MJm2",
    "Rainfall_mm",
    "Snowfall_cm",
]

CATEGORICAL_COLS = [
    "Season",
    "Holiday",
    "Functioning_Day",
    "Library_Branch",
    "Top_Category",
    "Membership_Type",
    "Day_of_Week",
]


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the rentals CSV into a DataFrame.

    Parameters
    ----------
    csv_path : str or Path
        Path to the rentals CSV file.

    Returns
    -------
    pd.DataFrame
        Raw rentals data.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. See scripts/download_data.sh")

    df = pd.read_csv(path)
    logger.info("Loaded %s rows × %s columns from %s", *df.shape, path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning steps to the rentals DataFrame.

    Steps:
    - Drop duplicate rows
    - Drop rows where the library was not functioning
    - Fill numeric missing values with the column median
    - Standardize Yes/No casing
    - Drop negative rental counts (clearly errors)
    - Parse Date column

    Parameters
    ----------
    df : pd.DataFrame
        Raw rentals data.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df = df.copy()

    # Standardize Yes/No first so case-variant duplicates collapse correctly
    for col in ("Holiday", "Functioning_Day"):
        df[col] = df[col].astype(str).str.strip().str.title()

    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %s duplicate rows", before - len(df))

    # Drop non-functioning days (no rentals possible anyway)
    df = df[df["Functioning_Day"] == "Yes"].copy()

    # Drop negative or null rentals
    df = df[df["Rentals_Count"].notna() & (df["Rentals_Count"] >= 0)].copy()

    # Fill numeric NaN with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    logger.info("Cleaned data: %s rows remaining", len(df))
    return df.reset_index(drop=True)
