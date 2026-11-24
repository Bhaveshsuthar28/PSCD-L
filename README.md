# TRINETRA — Persistent Scene Change Detection & Localization (PSCD&L)

TRINETRA is a full-stack surveillance video analytics system that detects **persistent visual scene changes**, tracks candidate regions over time, evaluates temporal persistence and optical-flow settling criteria, and produces auditable visual evidence and reports.

> **Important Scientific Scope:** TRINETRA detects persistent visual scene changes. It does **not** independently establish object identity, ownership, abandonment, or intent. Heuristic confidence scores reflect temporal and spatial persistence, not calibrated probabilities.

## Core Capabilities

- **Frozen Spatial Model (Task 1)**: DINOv2 ViT-B/14 backbone + trained change-detection decoder (`outputs/training/checkpoints/trinetra-33-0.1482.ckpt`, SHA-256 verified before loading).
- **Temporal Persistence Pipeline (Task 2)**: Candidate extraction, IoU/distance tracking, missed-observation handling, Farneback optical-flow settling analysis, monotonic lifecycle (`CANDIDATE -> OBSERVING -> SETTLING -> CONFIRMED -> COMPLETED`), post-confirmation degradation review flags, and evidence generation (`src/inference/`).
- **Integrated Backend (Task 3)**: FastAPI + SQLAlchemy + SQLite (WAL), bounded background job queue (4 GB VRAM-safe single-model residency), cooperative cancellation, restart recovery, SSE live streaming, HTTP range video streaming, secure evidence delivery, database analytics, and JSON/CSV/ZIP exports (`backend/`).
- **Operational Dashboard (Tasks 4 & 5)**: React 19 + TypeScript + Vite + Tailwind CSS v4 + Apache ECharts dual-theme (`light` / `dark` / `system`) UI with Video Library, Processing Console, Events Explorer, Evidence Viewer with 3-way comparison, Analytics & Exports, System Diagnostics, and Settings (`frontend/`).

## Technology Stack

- **AI / CV**: Python 3.11, PyTorch (`cu130`), Transformers (DINOv2), OpenCV, NumPy
- **Backend**: FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2, SQLite
- **Frontend**: React 19, TypeScript, Vite 6, Tailwind CSS v4, TanStack Query, React Router, React Hook Form + Zod, Apache ECharts, Lucide Icons
- **Testing**: pytest, httpx, Vitest, React Testing Library, Playwright

## Prerequisites

- Windows 10/11 (or Linux)
- Python 3.11+ (`.venv` virtual environment)
- Node.js 22+ and npm
- Frozen checkpoint at `outputs/training/checkpoints/trinetra-33-0.1482.ckpt`

## Quick Start (Windows PowerShell)

1. **Install & initialize backend**:
   ```powershell
   Set-Location C:\PSCD&L
   .\.venv\Scripts\Activate.ps1
   pip install -r backend/requirements.txt
   python -c "from backend.app.database.connection import create_database_manager; db = create_database_manager(); db.initialize(); db.dispose()"
   ```
2. **Install frontend**:
   ```powershell
   Set-Location C:\PSCD&L\frontend
   npm install
   ```
3. **Start Backend (Terminal 1, from repo root)**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```
4. **Start Frontend (Terminal 2, from `frontend/`)**:
   ```powershell
   Set-Location C:\PSCD&L\frontend
   npm run dev
   ```
5. **Access**:
   - Dashboard: `http://127.0.0.1:5173`
   - API Swagger Docs: `http://127.0.0.1:8000/docs`
   - Health Endpoint: `http://127.0.0.1:8000/health`

## Testing Commands

```powershell
# Backend + Task 2 regression + real ABODA GPU test
.\.venv\Scripts\python.exe -m pytest -v tests/ backend/tests/

# Frontend build, typecheck, unit/API tests, and full-stack Playwright E2E
Set-Location frontend
npm run build
npm run test
npm run test:e2e
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API.md) & [Backend API Contract](docs/BACKEND_API.md)
- [Configuration Reference](docs/CONFIGURATION.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Backend Architecture](docs/BACKEND_ARCHITECTURE.md) & [Database Schema](docs/DATABASE_SCHEMA.md)
- [Processing Pipeline](docs/PROCESSING_PIPELINE.md), [Error Handling](docs/ERROR_HANDLING.md), & [Security Policy](docs/SECURITY.md)
- [Frontend Architecture](docs/FRONTEND_ARCHITECTURE.md), [UI Design System](docs/UI_DESIGN_SYSTEM.md), [Frontend API Integration](docs/FRONTEND_API_INTEGRATION.md), & [Testing Guide](docs/FRONTEND_TESTING.md)
