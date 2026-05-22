"""Tests for src.data."""

import pandas as pd

from src.data import clean_data


def _sample_df():
    return pd.DataFrame(
        {
            "Date": ["01/01/2023", "01/01/2023", "02/01/2023", "03/01/2023", "04/01/2023"],
            "Hour": [9, 9, 10, 11, 12],
            "Rentals_Count": [50, 50, -1, None, 40],
            "Temperature_C": [22.0, 22.0, 25.0, None, 30.0],
            "Humidity_pct": [40.0, 40.0, 55.0, 60.0, None],
            "Wind_Speed_ms": [5.0, 5.0, 4.0, 3.0, 2.0],
            "Visibility_m": [1500.0, 1500.0, 1400.0, 1300.0, 1200.0],
            "Solar_Radiation_MJm2": [2.5, 2.5, 2.0, 1.5, 1.0],
            "Rainfall_mm": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Snowfall_cm": [0, 0, 0, 0, 0],
            "Season": ["Winter"] * 5,
            "Holiday": ["yes", "Yes", "no", "No", "Yes"],
            "Functioning_Day": ["Yes", "Yes", "Yes", "Yes", "No"],
            "Library_Branch": ["A", "A", "B", "C", "D"],
            "Top_Category": ["Fiction"] * 5,
            "Membership_Type": ["Regular"] * 5,
            "Day_of_Week": ["Sunday"] * 5,
        }
    )


def test_clean_data_drops_duplicates():
    df = _sample_df()
    cleaned = clean_data(df)
    # Duplicate row 0/1 should be collapsed
    assert cleaned.duplicated().sum() == 0


def test_clean_data_drops_negative_rentals():
    df = _sample_df()
    cleaned = clean_data(df)
    assert (cleaned["Rentals_Count"] >= 0).all()


def test_clean_data_drops_non_functioning_days():
    df = _sample_df()
    cleaned = clean_data(df)
    assert (cleaned["Functioning_Day"] == "Yes").all()


def test_clean_data_fills_numeric_missing():
    df = _sample_df()
    cleaned = clean_data(df)
    assert cleaned["Humidity_pct"].isna().sum() == 0
    assert cleaned["Temperature_C"].isna().sum() == 0


def test_clean_data_standardizes_yes_no():
    df = _sample_df()
    cleaned = clean_data(df)
    assert set(cleaned["Holiday"].unique()).issubset({"Yes", "No"})


def test_clean_data_parses_dates():
    df = _sample_df()
    cleaned = clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
