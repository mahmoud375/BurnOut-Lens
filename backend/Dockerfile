# ── BurnOut Lens — Backend Dockerfile ─────────────────────────────────────
#
# Build context: REPO ROOT (set in docker-compose.yml via context: .)
# This is required because the backend depends on ml/ (source + model).
#
# Single stage (not multi-stage) — rationale:
#   uv installs directly into the system python inside the container.
#   A multi-stage build for a pure-Python app with uv does not meaningfully
#   reduce the final image size because the installed packages (numpy, xgboost,
#   shap, etc.) are all needed at runtime — nothing gets "compiled away".
#   The only savings would be removing uv itself (~15MB), not worth the extra
#   Dockerfile complexity.

FROM python:3.12-slim

# Keeps Python from buffering stdout/stderr (important for container logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /app

# ── Install uv ────────────────────────────────────────────────────────────
# Pinned version for reproducibility.  The binary is self-contained.
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /usr/local/bin/uv

# ── Copy ml/ package (source + trained model) ─────────────────────────────
# ml/src/ is the Python package; ml/models/ contains the trained XGBoost
# artifact.  config.py derives MODEL_PATH as:
#   Path(__file__).parent.parent / "models" / "xgb_burnout_model.json"
# so the directory layout ml/src/ + ml/models/ must be preserved relative
# to each other inside the image.
COPY ml/ /app/ml/

# ── Copy backend app + dependency manifests ───────────────────────────────
COPY backend/pyproject.toml backend/uv.lock /app/backend/

# ── Install dependencies (no dev extras, no editable venv cache) ──────────
# --no-dev: skip pytest/httpx
# --no-cache: keep image lean
# --python python3: use the system python (already 3.12 in this image)
# uv reads [tool.uv.sources] burnout-lens = {path = "../ml", editable = true}
# The path resolves to /app/ml/ which we already copied above.
RUN uv sync \
        --project /app/backend \
        --no-dev \
        --no-cache \
        --python python3.12

# ── Copy backend application code ─────────────────────────────────────────
# Done after dependency install so code changes don't invalidate the
# (expensive) dependency layer.
COPY backend/app/ /app/backend/app/

WORKDIR /app/backend

EXPOSE 8000

CMD ["/app/backend/.venv/bin/uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
