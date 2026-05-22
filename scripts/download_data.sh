#!/usr/bin/env bash
# Placeholder for fetching the dataset.
# Replace the URL below with wherever you host the CSV (S3, GCS, Drive link, etc.)

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/jeddah_library_rentals.csv" ]; then
  echo "Dataset already present."
  exit 0
fi

echo "Place jeddah_library_rentals.csv into $DATA_DIR before running training."
echo "Or update this script with a real download URL."
