# Architecture & DevOps Decisions

This document walks through the design choices behind the project and maps them to the DevOps practices that motivated each one.

## 1. Code Organization

The notebook in `notebooks/` is the human-friendly walkthrough — useful for reviewers and stakeholders. All real logic lives in `src/` as importable Python modules. This separation means:

- The notebook stays a presentation layer (no business logic to maintain in two places).
- CI can lint, type-check, and unit-test `src/` without running the full notebook.
- The same modules power both the notebook and the headless `train.py` entrypoint.

## 2. Containerization (Docker)

Why Docker:
- **Reproducibility** — anyone can pull and run with the exact same Python and library versions.
- **No "works on my machine"** — TensorFlow and pandas can be finicky to install; the image handles it.
- **CI parity** — the same image runs in CI and on a developer laptop.

The `Dockerfile` is a single-stage Python 3.10 slim image. Layer caching is optimized by copying `requirements.txt` before the source.

## 3. CI/CD (GitHub Actions)

The pipeline in `.github/workflows/ci.yml` runs on every push and pull request:

1. **Lint** — `flake8` for style, `black --check` for formatting.
2. **Test** — `pytest` over the suite in `tests/`.
3. **Docker build** — verifies the image still builds (a fast smoke test).

Future extension: push the image to a registry, deploy a FastAPI inference service.

## 4. Configuration & Reproducibility

- `requirements.txt` pins majors/minors of every dependency.
- `RANDOM_STATE=42` is set globally in `src/train.py` for sklearn and TensorFlow.
- The `Makefile` is the single source of truth for common commands.

## 5. Observability

`src/train.py` uses Python's `logging` module (not `print`) so log output can be filtered by level. The training run also writes `artifacts/metrics.json` — a structured artifact that can be picked up by experiment trackers (MLflow, W&B) or even just diffed across runs.

## 6. Mapping to the DevOps Roadmap

| Roadmap topic | Where it shows up here |
|---|---|
| Linux fundamentals | Bash helper script in `scripts/`, Makefile targets |
| Version control (Git) | Repo layout, `.gitignore`, branch-friendly structure |
| Programming (Python) | Refactored modules under `src/` |
| Containerization (Docker) | `Dockerfile`, `docker-compose.yml` |
| CI/CD | `.github/workflows/ci.yml` |
| Configuration management | Declarative configs (`requirements.txt`, Compose) |
| Monitoring & logging | Python `logging`, metrics JSON artifact |
