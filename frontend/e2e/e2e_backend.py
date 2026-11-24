"""Isolated temporary backend server launcher for Playwright E2E tests."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="trinetra_e2e_"))
    storage_root = tmp_dir / "storage"
    log_dir = tmp_dir / "logs"
    db_path = storage_root / "e2e_trinetra.db"

    os.environ["TRINETRA_APP_ENV"] = "testing"
    os.environ["TRINETRA_DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["TRINETRA_STORAGE_ROOT"] = str(storage_root)
    os.environ["TRINETRA_LOG_DIR"] = str(log_dir)
    os.environ["TRINETRA_MIN_FREE_DISK_BYTES"] = str(1024 * 1024)
    os.environ["TRINETRA_CORS_ORIGINS"] = '["http://127.0.0.1:5173","http://localhost:5173"]'

    from backend.app.core.config import get_settings
    from backend.app.main import create_app

    get_settings.cache_clear()
    app = create_app(get_settings())
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
