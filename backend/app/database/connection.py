"""Database engine, session management, and SQLite pragmas."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.database.migrations import apply_migrations
from backend.app.database.models import Base


logger = get_logger("database")


def _configure_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
    finally:
        cursor.close()


class DatabaseManager:
    """Manages SQLAlchemy Engine, SessionLocal, and schema initialization."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        connect_args: dict = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            self._ensure_sqlite_parent(database_url)

        self.engine: Engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            future=True,
        )

        if database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", _configure_sqlite_pragmas)

        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            class_=Session,
        )

    @staticmethod
    def _ensure_sqlite_parent(database_url: str) -> None:
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            return
        if database_url.startswith("sqlite:///"):
            raw_path = database_url[len("sqlite:///"):]
            if raw_path and raw_path != ":memory:":
                Path(raw_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Create tables and apply pending schema migrations."""
        Base.metadata.create_all(bind=self.engine)
        with self.session() as db:
            apply_migrations(db)
        logger.info("Database initialized", extra={"operation": "db_init"})

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional database session with rollback on errors."""
        db = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def check_connectivity(self) -> bool:
        """Verify database connection without mutating state."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(
                "Database connectivity check failed: %s",
                exc,
                extra={"operation": "db_health", "error_code": "DATABASE_UNAVAILABLE"},
            )
            return False

    def dispose(self) -> None:
        self.engine.dispose()


def create_database_manager(settings: Settings | None = None) -> DatabaseManager:
    cfg = settings or get_settings()
    return DatabaseManager(cfg.database_url)
