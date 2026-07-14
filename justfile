# Task runner for common dev workflows. Install `just`: https://just.systems
# (Windows: `winget install Casey.Just` or `scoop install just`.)
# All recipes go through the uv-locked env, so versions match CI.

# list available recipes
default:
    @just --list

# create the venv, install deps + dev tools, and the git hooks
setup:
    uv sync
    uv run --no-sync pre-commit install
    uv run --no-sync pre-commit install --hook-type pre-push

# lint + format-check (no changes)
lint:
    uv run --no-sync ruff check .
    uv run --no-sync ruff format --check .

# auto-fix lint + format in place
fmt:
    uv run --no-sync ruff format .
    uv run --no-sync ruff check --fix .

# static type-check (app/ + eval/)
typecheck:
    uv run --no-sync mypy

# run the test suite (pass extra args, e.g. `just test -k library`)
test *ARGS:
    uv run --no-sync python -m pytest {{ARGS}}

# tests with coverage (enforces the fail-under floor)
cov:
    uv run --no-sync python -m pytest --cov=app --cov=eval --cov-report=term-missing

# everything CI runs
check: lint typecheck test

# audit dependencies for known vulnerabilities
audit:
    uv run --no-sync pip-audit

# run the Streamlit app
serve:
    uv run streamlit run streamlit_app.py

# --- offline data pipeline (needs the torch env) ---

# build the full 10k dataset (books + embeddings + EASE CF)
build-data:
    uv run --no-sync python scripts/build_real_dataset.py

# fine-tune the co-read encoder, then re-embed the catalog with it
finetune:
    uv run --no-sync python scripts/finetune_coread.py --steps 60
    uv run --no-sync python scripts/build_embeddings.py

# run the paradigm scoreboard (popularity / content / EASE / hybrid)
eval:
    uv run --no-sync python -m eval.compare_paradigms

# build + run the minimal serving container
docker:
    docker build -t book-recommender .
    docker run --rm -p 8501:8501 book-recommender
