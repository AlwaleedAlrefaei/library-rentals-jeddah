# 📚 Library Rentals Demand Prediction — Jeddah

[![CI](https://github.com/YOUR_USERNAME/library-rentals-jeddah/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/library-rentals-jeddah/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> End-to-end machine learning project predicting **hourly book rental demand** across library branches in Jeddah, Saudi Arabia. Built with reproducibility and DevOps best practices in mind.

---

## 🎯 Project Overview

A chain of public libraries in Jeddah needs to optimize **staffing and inventory** based on predicted hourly rentals. This project takes a messy CSV export and turns it into a production-style ML pipeline that:

1. **Cleans and explores** ~6,600 rows of rental, weather, and membership data
2. **Engineers features** like peak hours, weekend flags, and temperature bins
3. **Trains and compares 4 models**: Linear Regression, Decision Tree, Random Forest, and a Neural Network (Keras)
4. **Reports metrics** (R², MAE, RMSE) and exports the winning model

## 🏆 Results

Trained on 6,264 rows after cleaning, 80/20 train-test split, `random_state=42`.

| Model              | R²        | MAE      | RMSE     |
|--------------------|-----------|----------|----------|
| Linear Regression  | 0.845     | 6.55     | 8.30     |
| Decision Tree      | 0.781     | 7.40     | 9.87     |
| **Random Forest**  | **0.892** | **5.21** | **6.94** |
| Neural Network     | 0.873     | 5.77     | 7.53     |

> 🏅 **Random Forest won** — it captures non-linear interactions (hour × season × branch) without overfitting like the single Decision Tree, and edges out the Neural Network on this dataset size.

---

## 🗂 Repository Structure

```
library-rentals-jeddah/
├── .github/workflows/      # GitHub Actions CI pipeline
│   └── ci.yml
├── notebooks/              # Jupyter notebook (full walkthrough)
│   └── library_rentals_analysis.ipynb
├── src/                    # Reusable Python modules
│   ├── data.py             # Data loading & cleaning
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model definitions
│   └── train.py            # Training entrypoint
├── tests/                  # Unit tests (pytest)
│   ├── test_data.py
│   └── test_features.py
├── scripts/                # Helper scripts
│   └── download_data.sh
├── docs/                   # Architecture & design docs
│   └── architecture.md
├── Dockerfile              # Containerized environment
├── docker-compose.yml      # One-command Jupyter + training
├── Makefile                # Reproducible commands
├── requirements.txt        # Pinned Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)

```bash
# Build and run Jupyter
docker compose up jupyter

# Or train the model headlessly
docker compose run --rm train
```

### Option 2 — Local Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python -m src.train --data data/jeddah_library_rentals.csv
```

### Option 3 — Make

```bash
make install     # install deps
make test        # run unit tests
make train       # run training pipeline
make notebook    # launch Jupyter
make lint        # run linters
make docker      # build Docker image
```

---

## 🛠 DevOps Practices Applied

This project intentionally borrows from the DevOps engineering toolkit:

| Practice | Tool | Where to look |
|---|---|---|
| **Version Control** | Git + `.gitignore` | Repository structure, branch-friendly layout |
| **Containerization** | Docker + Compose | `Dockerfile`, `docker-compose.yml` |
| **CI/CD** | GitHub Actions | `.github/workflows/ci.yml` — lints + tests on every push |
| **Infrastructure as Code** | Declarative configs | `requirements.txt`, `Makefile`, Compose file |
| **Automation** | Python scripts + Make | `scripts/`, `Makefile` |
| **Monitoring/Logging** | Python `logging` + metrics JSON | `src/train.py` emits structured logs and a metrics artifact |
| **Reproducibility** | Pinned deps + random seeds | `requirements.txt`, seed=42 throughout |

---

## 📊 Dataset

The dataset `jeddah_library_rentals.csv` contains 6,609 hourly records with columns including:

- **Target**: `Rentals_Count` (books rented per hour)
- **Weather**: `Temperature_C`, `Humidity_pct`, `Wind_Speed_ms`, `Visibility_m`, `Solar_Radiation_MJm2`, `Rainfall_mm`, `Snowfall_cm`
- **Context**: `Date`, `Hour`, `Season`, `Holiday`, `Functioning_Day`, `Day_of_Week`
- **Library**: `Library_Branch`, `Top_Category`, `Membership_Type`

> 📥 Place the CSV in `data/jeddah_library_rentals.csv` before running. The `data/` directory is gitignored.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Unit tests cover:
- Data cleaning (missing values, duplicates, negative counts)
- Feature engineering (peak hours, weekend flag, temperature bins)

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers GitHub Actions to:

1. ✅ Lint with `flake8` and check formatting with `black`
2. ✅ Run unit tests with `pytest`
3. ✅ Build the Docker image to verify it still builds

See `.github/workflows/ci.yml`.

---

## 📈 Next Steps

- [ ] Add MLflow for experiment tracking
- [ ] Deploy the model as a FastAPI service
- [ ] Add Prometheus metrics endpoint for inference monitoring
- [ ] Terraform module to provision the deployment on AWS

---

## 📝 License

MIT — see [LICENSE](LICENSE).