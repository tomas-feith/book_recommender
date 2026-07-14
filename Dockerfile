# Minimal serving image. The app serves with numpy + scipy + streamlit only --
# torch / sentence-transformers are OFFLINE-ONLY (used to build embeddings), so the
# runtime image stays small and never loads a model.
#
# Prerequisite: build the (gitignored, regenerable) data artifacts first, so they
# are in the build context:
#     just build-data          # -> data/real_books.json, real_embeddings.npz, real_cf.npz
# Then:
#     docker build -t book-recommender .
#     docker run --rm -p 8501:8501 book-recommender

FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Serving-only dependency set (no torch). Kept explicit so the image is lean.
RUN uv pip install --system --no-cache \
    "numpy>=1.26" "scipy>=1.18" "streamlit>=1.57" "openpyxl>=3.1"

# Application code + the offline-built data artifacts
COPY app ./app
COPY eval/__init__.py eval/data.py ./eval/
COPY .streamlit ./.streamlit
COPY streamlit_app.py ./
COPY data/real_books.json data/real_embeddings.npz data/real_cf.npz ./data/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
