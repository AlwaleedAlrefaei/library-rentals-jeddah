.PHONY: help install test lint format train notebook docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	pip install -r requirements.txt
	pip install flake8 black pytest

test:  ## Run unit tests
	pytest tests/ -v

lint:  ## Run linters
	flake8 src tests --max-line-length=100 --extend-ignore=E203,W503
	black --check src tests --line-length=100

format:  ## Auto-format code
	black src tests --line-length=100

train:  ## Run the training pipeline
	python -m src.train --data data/jeddah_library_rentals.csv

notebook:  ## Launch Jupyter
	jupyter notebook notebooks/

docker:  ## Build the Docker image
	docker build -t library-rentals-jeddah:latest .

docker-train:  ## Run training inside Docker
	docker compose run --rm train

clean:  ## Remove caches and artifacts
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf artifacts/*.json
