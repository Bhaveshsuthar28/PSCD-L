# TRINETRA Backend

FastAPI backend for **TRINETRA — Persistent Scene Change Detection and Localization (PSCD&L)**.

See root [README.md](../README.md) and [docs/BACKEND_API.md](../docs/BACKEND_API.md) for complete setup, API contracts, database schema, security policy, and testing commands.

## Quick Start (Windows PowerShell from repo root)

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
