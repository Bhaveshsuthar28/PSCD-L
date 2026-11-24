"""Lightweight versioned schema migrations for TRINETRA database."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import SchemaMigration, SystemSetting


MIGRATIONS: list[tuple[str, str, callable]] = []


def _v001_seed_default_settings(db: Session) -> None:
    defaults = [
        (
            "retention.auto_delete_evidence",
            False,
            "Never automatically delete confirmed event evidence.",
        ),
        (
            "processing.default_sampling_interval_seconds",
            1.0,
            "Default temporal sampling interval in seconds.",
        ),
        (
            "system.notice",
            "TRINETRA detects persistent visual scene changes and does not independently establish object identity, ownership, abandonment, or intent.",
            "Mandatory scientific interpretation notice.",
        ),
    ]
    for key, value, desc in defaults:
        existing = db.get(SystemSetting, key)
        if existing is None:
            db.add(SystemSetting(key=key, value_json=value, description=desc))


MIGRATIONS.append(("001_initial_schema_and_settings", "Initial TRINETRA v1 schema and default non-secret settings", _v001_seed_default_settings))


def apply_migrations(db: Session) -> list[str]:
    """Apply any unrecorded migrations idempotently."""
    applied_versions = set(
        db.scalars(select(SchemaMigration.version)).all()
    )
    newly_applied: list[str] = []
    for version, description, fn in MIGRATIONS:
        if version in applied_versions:
            continue
        fn(db)
        db.add(SchemaMigration(version=version, description=description))
        newly_applied.append(version)
    db.flush()
    return newly_applied
