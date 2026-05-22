"""Tests for src.features."""

import pandas as pd

from src.features import (
    add_peak_hour_flag,
    add_temperature_bin,
    add_time_features,
    add_weekend_flag,
    engineer_features,
)


def _df():
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-15", "2023-06-20", "2023-12-25"]),
            "Hour": [9, 17, 23],
            "Temperature_C": [10.0, 28.0, 37.0],
            "Day_of_Week": ["Sunday", "Friday", "Monday"],
        }
    )


def test_add_time_features_creates_month_and_day():
    out = add_time_features(_df())
    assert out["Month"].tolist() == [1, 6, 12]
    assert out["Day"].tolist() == [15, 20, 25]


def test_peak_hour_flag():
    out = add_peak_hour_flag(_df())
    # Hour 9 not peak, 17 peak, 23 not peak
    assert out["Is_Peak_Hour"].tolist() == [0, 1, 0]


def test_weekend_flag_uses_saudi_weekend():
    out = add_weekend_flag(_df())
    # Friday is weekend in Saudi Arabia
    assert out["Is_Weekend"].tolist() == [0, 1, 0]


def test_temperature_bin_categories():
    out = add_temperature_bin(_df())
    assert out["Temperature_Bin"].tolist() == ["Cold", "Warm", "Hot"]


def test_engineer_features_runs_full_pipeline():
    out = engineer_features(_df())
    for col in ("Month", "Day", "Is_Peak_Hour", "Is_Weekend", "Temperature_Bin"):
        assert col in out.columns
