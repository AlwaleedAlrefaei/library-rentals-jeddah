"""Model definitions for the rentals demand prediction project."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Evaluation metrics for a trained model."""

    name: str
    r2: float
    mae: float
    rmse: float

    def to_dict(self) -> dict:
        return {"model": self.name, "r2": self.r2, "mae": self.mae, "rmse": self.rmse}


def build_linear_regression() -> LinearRegression:
    return LinearRegression()


def build_decision_tree(random_state: int = 42) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(max_depth=12, random_state=random_state)


def build_random_forest(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        n_jobs=-1,
        random_state=random_state,
    )


def build_neural_network(input_dim: int):
    """Build a small Keras MLP regressor.

    Imported lazily so the rest of the package works without TensorFlow.
    """
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelResult:
    """Compute R², MAE, and RMSE."""
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    result = ModelResult(name=name, r2=r2, mae=mae, rmse=rmse)
    logger.info("%s — R²=%.3f  MAE=%.2f  RMSE=%.2f", name, r2, mae, rmse)
    return result
