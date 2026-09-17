# BurnOut Lens

> Predict employee burnout risk (0–10) with Explainable AI (SHAP).

![CI](https://github.com/mahmoud375/BurnOut-Lens/actions/workflows/ci.yml/badge.svg)![CD](https://github.com/mahmoud375/BurnOut-Lens/actions/workflows/cd.yml/badge.svg)![Live Demo](https://img.shields.io/badge/Live_Demo-burnoutlens.tech-success?style=flat&logo=nginx)![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)![React 19](https://img.shields.io/badge/React-19-61dafb.svg)---

## Overview

BurnOut Lens predicts an employee's burnout risk on a continuous **0.0–10.0** scale, mapped to 4 severity tiers (**Low**, **Moderate**, **High**, **Severe**), using a machine learning model trained on 100,000 tech-worker profiles. Every prediction is paired with a **SHAP** breakdown showing exactly which habits, pressures, or protective factors drove the score — turning a black-box number into an actionable explanation.

---

## Live Demo

- **Web Application**: <https://burnoutlens.tech>

---

## How It Works

### The Prediction

The model accepts 9 workplace and lifestyle factors:

1. **Seniority Level**: `Junior`, `Mid`, `Senior`, `Lead`, `Manager`, or `Principal`.
2. **Weekly Work Hours**: `20.0`–`90.0`.
3. **Meetings Per Day**: `0.0`–`15.0`.
4. **Sleep Per Night**: `2.0`–`12.0` hours.
5. **Exercise Days**: `0.0`–`7.0` days/week.
6. **Vacation Days Taken**: `0.0`–`30.0` per year.
7. **Social Support Score**: `1.0`–`10.0`.
8. **Manager Support Score**: `1.0`–`10.0`.
9. **Deadline Pressure Score**: `1.0`–`10.0`.

The API returns:

- `burnout_score` — predicted score, clipped to `[0.0, 10.0]`.
- `burnout_level` — `Low` / `Moderate` / `High` / `Severe`.
- `base_value` — dataset-wide average prediction baseline (\~5.40).
- `contributions` — all 9 features ranked by absolute SHAP impact.

### The Explainability Angle (SHAP)

A raw score of `7.4` tells someone they're at high risk but not *why*. SHAP attributes each prediction to its inputs:

- **Positive (**`increases`**)** — pushes the score up (e.g. `+0.85` from a 52-hour work week).
- **Negative (**`decreases`**)** — acts as a protective buffer (e.g. `-0.54` from 8 hours of sleep).

---

## Tech Stack

| Layer | Technology | Version | Purpose |
| --- | --- | --- | --- |
| **Frontend** | React | `19.2.8` | Component-driven user interface |
|  | Vite | `8.3.0` | Development server and static asset bundler |
|  | TypeScript | `~6.0.2` | Static typing and API contract enforcement |
|  | Oxlint | `^1.81.0` | Fast JavaScript/TypeScript linting |
|  | Nginx | `1.27-alpine` | Production static asset container server |
| **Backend** | FastAPI | `>=0.115.0` | High-performance asynchronous REST API framework |
|  | Uvicorn | `>=0.32.0` | ASGI production application server |
|  | Python | `3.12` | Core backend runtime |
|  | Pydantic | `>=2.10.0` | Request validation and response serialization |
|  | Pydantic Settings | `>=2.7.0` | Environment variable management |
|  | uv | `0.12.0` | Ultra-fast Python package and venv manager |
| **ML & Explainability** | XGBoost | `3.4.1` (`xgboost-cpu`) | Gradient-boosted regression model |
|  | SHAP | `>=0.47.2` | TreeExplainer for exact Shapley feature attribution |
|  | scikit-learn | `>=1.9.1` | Train/test splitting and evaluation metrics |
|  | pandas | `>=3.0.5` | Tabular data manipulation |
|  | pyarrow | `>=18.0` | High-performance columnar dataset processing |
| **Infra & DevOps** | Docker | Engine & Compose | Multi-container isolation and packaging |
|  | GitHub Actions | CI & CD | Automated testing, linting, image builds, and deployment |
|  | Docker Hub | Registry | Public container image storage |
|  | Host Nginx | Reverse Proxy | SSL/TLS termination, rate limiting, and security headers |
|  | Let's Encrypt | Certbot | Automated HTTPS certificates |
|  | Azure VPS | Ubuntu Linux | Cloud hosting infrastructure |

---

## Machine Learning

### Model & Hyperparameters

- **Algorithm**: `XGBRegressor`, histogram-based tree splitting (`tree_method: "hist"`).
- **Dataset**: `mental_health_burnout_tech_2026.csv` — 100,000 synthetic tech-worker records.
- **Split**: 80% train (80,000) / 20% test (20,000), fixed seed (`42`).
- **Hyperparameters**:

  ```python
  {
      "n_estimators": 300,
      "max_depth": 5,
      "learning_rate": 0.05,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "tree_method": "hist",
      "random_state": 42
  }
  ```

### Performance (Held-Out Test Set)

- **R²**: `0.6987` (\~70% of variance explained)
- **MAE**: `1.1641` points
- **RMSE**: `1.4619` points

### Explainability Engine

- `shap.TreeExplainer` computes exact Shapley values via tree traversal (no sampling approximation).
- Model (`xgb_burnout_model.json`, \~1.08 MB) and explainer are loaded once at startup as a singleton in `backend/app/core/model_loader.py` (\~110ms model load, \~88ms explainer build) — zero reload overhead per request.

### Backend Packaging

`ml/` is a standalone package (`burnout-lens`), installed as an editable local dependency in the backend (`burnout-lens = { path = "../ml", editable = true }`), so training logic, feature bounds, and inference share one source of truth.

---

## API Reference

### Endpoints

| Method | Endpoint | Description | Auth |
| --- | --- | --- | --- |
| `GET` | `/health` | Liveness check and model readiness | None |
| `POST` | `/predict` | Prediction + SHAP explanation | None |
| `GET` | `/` | Redirects to Swagger UI | None |
| `GET` | `/docs` | OpenAPI Swagger UI | None |
| `GET` | `/redoc` | ReDoc UI | None |
| `GET` | `/openapi.json` | OpenAPI 3.1 schema | None |

### `POST /predict` Example

**Request** (`POST /predict` or `POST /api/predict`):

```json
{
  "seniority_level": "Senior",
  "work_hours_per_week": 48.0,
  "meetings_per_day": 5.0,
  "sleep_hours_per_night": 6.5,
  "exercise_days_per_week": 2.0,
  "vacation_days_taken": 8.0,
  "social_support_score": 6.0,
  "manager_support_score": 7.0,
  "deadline_pressure_score": 7.5
}
```

**Response** (`200 OK`):

```json
{
  "burnout_score": 6.82,
  "burnout_level": "High",
  "base_value": 5.40,
  "contributions": [
    { "feature": "deadline_pressure_score", "value": 7.5, "shap_value": 0.84, "direction": "increases" },
    { "feature": "work_hours_per_week", "value": 48.0, "shap_value": 0.65, "direction": "increases" },
    { "feature": "vacation_days_taken", "value": 8.0, "shap_value": -0.42, "direction": "decreases" },
    { "feature": "manager_support_score", "value": 7.0, "shap_value": -0.31, "direction": "decreases" },
    { "feature": "meetings_per_day", "value": 5.0, "shap_value": 0.28, "direction": "increases" },
    { "feature": "sleep_hours_per_night", "value": 6.5, "shap_value": -0.22, "direction": "decreases" },
    { "feature": "social_support_score", "value": 6.0, "shap_value": -0.18, "direction": "decreases" },
    { "feature": "exercise_days_per_week", "value": 2.0, "shap_value": -0.12, "direction": "decreases" },
    { "feature": "seniority_level", "value": 2.0, "shap_value": 0.08, "direction": "increases" }
  ]
}
```

---

## Architecture & Deployment

### Deployment Pipeline Overview

```mermaid
flowchart TD
    subgraph GitHub ["GitHub Repository"]
        A[git push / PR] --> B[CI Workflow]
        B --> C{CI Success on main?}
        C -- Yes --> D[CD Workflow]
    end

    subgraph CI ["CI Pipeline (.github/workflows/ci.yml)"]
        B1[ml-tests]
        B2[backend-tests]
        B3[lint: ruff]
        B4[frontend-build: tsc + vite]
        B5[docker-build-check]
    end

    B -.->|runs| B1
    B -.->|runs| B2
    B -.->|runs| B3
    B -.->|runs| B4
    B -.->|runs| B5

    subgraph CD ["CD Pipeline (.github/workflows/cd.yml)"]
        D --> D1[docker-push: Build & Push Images]
        D1 --> D2[Deploy via SSH Action]
    end

    subgraph Registries ["Docker Hub Registry"]
        D1 --> R1["elgendy2003/burnout-lens-backend:latest"]
        D1 --> R2["elgendy2003/burnout-lens-frontend:latest"]
    end

    subgraph VPS ["Azure VPS (Ubuntu Linux)"]
        D2 --> S1[SSH Commands]
        S1 --> S2[docker-compose pull]
        S2 --> S3[docker-compose down]
        S3 --> S4[docker-compose up -d]
        S4 --> S5[docker image prune -f]

        subgraph Containers ["Docker Runtime (127.0.0.1)"]
            C_FE["Frontend Container\n(127.0.0.1:8080)"]
            C_BE["Backend Container\n(127.0.0.1:8000)"]
        end

        subgraph Ingress ["Host Nginx (Reverse Proxy + SSL)"]
            NX[Nginx 443 / 80]
            NX -- "location /api/" --> C_BE
            NX -- "location /" --> C_FE
        end
    end

    R1 -->|"pulled directly"| S2
    R2 -->|"pulled directly"| S2

    Internet([Client Browser]) -->|HTTPS: burnoutlens.tech| NX
```

### Continuous Integration (CI)

`.github/workflows/ci.yml` runs 5 parallel jobs on every push and PR, across all branches:

1. `ml-tests` — 128 tests over feature engineering, bounds, and model logic.
2. `backend-tests` — 55 tests over API endpoints and validation, resolving `ml/` as an editable dependency.
3. `lint` — `ruff check` on `ml/` and `backend/`.
4. `frontend-build` — `tsc -b` type check + production Vite build.
5. `docker-build-check` — builds both Dockerfiles without pushing, as a sanity check.

### Continuous Deployment (CD)

`.github/workflows/cd.yml` triggers via `workflow_run` only after CI succeeds on `main`:

1. `docker-push` — builds and pushes both images (`linux/amd64`) to Docker Hub, tagged `latest` and the 7-character commit SHA. Frontend is built with `VITE_API_BASE_URL=/api` for same-origin routing.
2. `deploy` — connects to the Azure VPS over SSH as a dedicated `deploy` user and runs:

   ```bash
   docker-compose -f docker-compose.prod.yml pull
   docker-compose -f docker-compose.prod.yml down
   docker-compose -f docker-compose.prod.yml up -d
   docker image prune -f
   ```

   This causes a brief (few-second) service interruption per deploy — not zero-downtime.

### Host Nginx & HTTPS

- Listens on `80`/`443` for `burnoutlens.tech`; `location /api/` proxies to the backend, `location /` to the frontend.
- Backend and frontend containers are bound to `127.0.0.1` only — never exposed directly to the internet.
- Certificates managed via Let's Encrypt/Certbot with automatic HTTP→HTTPS redirect.

### Published Container Images

- **Backend**: `elgendy2003/burnout-lens-backend`
- **Frontend**: `elgendy2003/burnout-lens-frontend`

---

## Security & Hardening

Following a security and performance audit of the production deployment, the following measures were implemented:

| Measure | Implementation |
| --- | --- |
| **Swap space** | 2 GB swap file added on the VPS (891 MB physical RAM) to prevent OOM kills under memory pressure |
| **Rate limiting** | Nginx `limit_req` on `/api/` — 10 req/min per IP, burst of 5 — protects the CPU-bound `/predict` endpoint from abuse |
| **Container resource limits** | `mem_limit`/`cpus` caps in `docker-compose.prod.yml` (backend: 400MB/0.5 CPU, frontend: 100MB/0.3 CPU) |
| **Security headers** | HSTS, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, and a restrictive CSP set at the Nginx layer |
| **Firewall** | `ufw` enabled with a default-deny policy; only ports `22`, `80`, `443` allowed |
| **fail2ban** | SSH brute-force protection — 5 failed attempts within 10 minutes triggers a 1-hour IP ban |
| **Log rotation** | Docker's `json-file` driver capped at 10MB × 3 files per container |
| **Non-root containers** | Backend runs as a dedicated `appuser`; frontend runs as Nginx's built-in `nginx` user on unprivileged port `8080` |

---

## Configuration

| Variable | Scope | Type | Purpose | Default / Example |
| --- | --- | --- | --- | --- |
| `CORS_ORIGINS` | Backend | Config | Comma-separated allowed browser origins | `http://localhost:5173,http://68.210.99.220,http://burnoutlens.tech,https://burnoutlens.tech` |
| `BURNOUT_MODEL_PATH` | Backend | Config | Path override for model JSON artifact | `ml/models/xgb_burnout_model.json` |
| `VITE_API_BASE_URL` | Frontend | Build Arg | API base URL baked into JS bundle | Local: `http://localhost:8000`<br>Prod: `/api` |
| `DOCKERHUB_USERNAME` | CI/CD | Secret | Docker Hub account username | `elgendy2003` |
| `DOCKERHUB_TOKEN` | CI/CD | Secret | Docker Hub Personal Access Token | `***` |
| `VPS_HOST` | CI/CD | Secret | Deployment server host/IP | `burnoutlens.tech` |
| `VPS_USERNAME` | CI/CD | Secret | Dedicated SSH deployment user | `deploy` |
| `VPS_SSH_KEY` | CI/CD | Secret | Private SSH key for the deploy user | `***` |

---

## Getting Started Locally

### Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- `uv`
- [Node.js 22+](https://nodejs.org/) & `npm`
- [Docker & Docker Compose](https://docs.docker.com/get-docker/) (optional)

### Option 1: Native Development

```bash
# Backend
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
# API at http://localhost:8000 (docs at /docs)
```

```bash
# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# App at http://localhost:5173
```

### Option 2: Full-Stack Docker Compose

```bash
docker compose up --build
# Frontend: http://localhost (port 80)
# Backend:  http://localhost:8000
```

### Option 3: Retraining the ML Model

```bash
cd ml
uv sync
uv run python -m src.train      # preprocess + train
uv run python -m src.evaluate   # metrics + SHAP plots
```

*Artifacts written to* `ml/models/xgb_burnout_model.json`*, metrics/plots to* `ml/reports/`*.*

---

## Project Structure

```
BurnOut Lens/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI: 5 parallel test/lint/build jobs
│       └── cd.yml                 # CD: Docker Hub publishing + SSH deploy to Azure VPS
├── backend/
│   ├── app/
│   │   ├── api/routes/            # API endpoints (/predict, /health)
│   │   ├── core/                  # Model & SHAP singleton loaders, configuration
│   │   ├── schemas/               # Pydantic schemas (BurnoutRequest, BurnoutResponse)
│   │   ├── services/               # Prediction and SHAP orchestration service
│   │   └── main.py                # FastAPI factory, CORS setup, root docs redirect
│   ├── tests/                     # 55 pytest tests covering endpoints & validation
│   ├── Dockerfile                 # Non-root Python 3.12-slim container definition
│   └── pyproject.toml             # Backend dependencies and tool configurations
├── frontend/
│   ├── public/                    # Static web assets (favicon.svg)
│   ├── src/
│   │   ├── api/                   # API client (predict.ts) reading VITE_API_BASE_URL
│   │   ├── components/            # BurnoutForm.tsx and ResultPanel.tsx with SHAP bars
│   │   ├── types/                 # TypeScript interfaces for API payloads
│   │   ├── App.tsx                # Main single-page application component
│   │   └── main.tsx               # React application entrypoint
│   ├── nginx.conf                 # Production Nginx SPA fallback config (non-root, port 8080)
│   ├── Dockerfile                 # Multi-stage, non-root container (Node 22 build -> Nginx 1.27 serve)
│   └── package.json               # Frontend dependencies and npm scripts
├── ml/
│   ├── data/                      # Raw dataset (100k rows) and processed parquet splits
│   ├── models/                    # Serialized XGBoost model (xgb_burnout_model.json)
│   ├── reports/                   # Training metrics (metrics.json) & SHAP charts
│   ├── src/                       # Preprocessing, config, training, evaluation, XAI
│   ├── tests/                     # 128 pytest tests for feature bounds, logic, and models
│   └── pyproject.toml             # ML package metadata (installable as burnout-lens)
├── docker-compose.yml             # Local development compose definition
├── docker-compose.prod.yml        # Production compose spec (Docker Hub images, resource limits)
└── README.md                      # Project documentation
```

---

## Testing

```bash
# ML pipeline — 128 tests
cd ml && uv run pytest -v

# Backend API — 55 tests
cd backend && uv run pytest -v

# Linting
uv tool run ruff check ml/src/ ml/tests/
uv tool run ruff check backend/app/ backend/tests/
```

---

## License

This project is licensed under the [MIT License](LICENSE).