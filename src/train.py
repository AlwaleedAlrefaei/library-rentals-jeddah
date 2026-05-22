"""End-to-end training entrypoint.

Run with:
    python -m src.train --data data/jeddah_library_rentals.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import clean_data, load_data
from src.features import engineer_features
from src.models import (
    build_decision_tree,
    build_linear_regression,
    build_neural_network,
    build_random_forest,
    evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

RANDOM_STATE = 42
TARGET = "Rentals_Count"
DROP_FOR_MODELING = ["Date", TARGET, "Functioning_Day"]


def prepare_xy(df: pd.DataFrame):
    """Encode categoricals and return X, y arrays."""
    y = df[TARGET].values
    X = df.drop(columns=DROP_FOR_MODELING, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rentals demand models")
    parser.add_argument(
        "--data",
        default="data/jeddah_library_rentals.csv",
        help="Path to the rentals CSV file",
    )
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Directory for trained models and metrics",
    )
    parser.add_argument(
        "--skip-nn",
        action="store_true",
        help="Skip the neural network (useful in CI)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df = load_data(args.data)
    df = clean_data(df)
    df = engineer_features(df)

    logger.info("Preparing features...")
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Linear Regression
    logger.info("Training Linear Regression...")
    lr = build_linear_regression()
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate("Linear Regression", y_test, lr.predict(X_test_scaled)))

    # Decision Tree
    logger.info("Training Decision Tree...")
    dt = build_decision_tree(RANDOM_STATE)
    dt.fit(X_train, y_train)
    results.append(evaluate("Decision Tree", y_test, dt.predict(X_test)))

    # Random Forest
    logger.info("Training Random Forest...")
    rf = build_random_forest(RANDOM_STATE)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

    # Neural Network (optional)
    if not args.skip_nn:
        logger.info("Training Neural Network...")
        import tensorflow as tf

        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        nn = build_neural_network(input_dim=X_train_scaled.shape[1])
        nn.fit(
            X_train_scaled,
            y_train,
            epochs=30,
            batch_size=64,
            verbose=0,
            validation_split=0.1,
        )
        nn_pred = nn.predict(X_test_scaled, verbose=0).flatten()
        results.append(evaluate("Neural Network", y_test, nn_pred))

    # Persist metrics
    metrics_path = output_dir / "metrics.json"
    payload = {"results": [r.to_dict() for r in results]}
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote metrics to %s", metrics_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"{'Model':<22} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("=" * 60)
    for r in results:
        print(f"{r.name:<22} {r.r2:>8.3f} {r.mae:>8.2f} {r.rmse:>8.2f}")
    print("=" * 60)

    best = max(results, key=lambda r: r.r2)
    print(f"\n🏆 Best model: {best.name} (R²={best.r2:.3f})")


if __name__ == "__main__":
    main()
