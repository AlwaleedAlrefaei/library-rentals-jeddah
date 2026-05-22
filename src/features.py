"""Feature engineering for the Jeddah library rentals dataset."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PEAK_HOURS = {10, 11, 16, 17, 18, 19}
WEEKEND_DAYS = {"Friday", "Saturday"}  # Saudi Arabian weekend


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Month and Day columns from the Date field."""
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    return df


def add_peak_hour_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary Is_Peak_Hour column based on typical busy hours."""
    df = df.copy()
    df["Is_Peak_Hour"] = df["Hour"].isin(PEAK_HOURS).astype(int)
    return df


def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary Is_Weekend column (Friday/Saturday in Saudi Arabia)."""
    df = df.copy()
    df["Is_Weekend"] = df["Day_of_Week"].isin(WEEKEND_DAYS).astype(int)
    return df


def add_temperature_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bin temperature into Cold / Mild / Warm / Hot categories."""
    df = df.copy()
    bins = [-100, 15, 25, 35, 100]
    labels = ["Cold", "Mild", "Warm", "Hot"]
    df["Temperature_Bin"] = pd.cut(df["Temperature_C"], bins=bins, labels=labels)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned rentals data.

    Returns
    -------
    pd.DataFrame
        Data enriched with engineered features.
    """
    logger.info("Engineering features...")
    df = add_time_features(df)
    df = add_peak_hour_flag(df)
    df = add_weekend_flag(df)
    df = add_temperature_bin(df)
    logger.info("Feature engineering complete. Columns: %s", list(df.columns))
    return df
