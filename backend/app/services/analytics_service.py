"""Database-backed analytics aggregation service."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.core.exceptions import ValidationError
from backend.app.database.connection import DatabaseManager
from backend.app.database.repositories import AnalyticsRepository


class AnalyticsService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    @staticmethod
    def _parse_date_range(
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[datetime | None, datetime | None]:
        parsed_start = None
        parsed_end = None
        try:
            if start_date:
                parsed_start = datetime.fromisoformat(start_date)
                if parsed_start.tzinfo is None:
                    parsed_start = parsed_start.replace(tzinfo=timezone.utc)
            if end_date:
                parsed_end = datetime.fromisoformat(end_date)
                if parsed_end.tzinfo is None:
                    parsed_end = parsed_end.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValidationError(
                "Invalid ISO 8601 date format in analytics filter.",
                error_code="INVALID_DATE_FILTER",
            ) from exc

        if parsed_start and parsed_end and parsed_end < parsed_start:
            raise ValidationError(
                "end_date cannot precede start_date.",
                error_code="INVALID_DATE_RANGE",
            )
        return parsed_start, parsed_end

    def overview(self) -> dict:
        with self.db_manager.session() as db:
            return AnalyticsRepository(db).overview()

    def videos(self) -> dict:
        with self.db_manager.session() as db:
            items = AnalyticsRepository(db).videos_breakdown()
            return {"items": items, "count": len(items)}

    def events(self, *, start_date: str | None = None, end_date: str | None = None) -> dict:
        s_dt, e_dt = self._parse_date_range(start_date, end_date)
        with self.db_manager.session() as db:
            return AnalyticsRepository(db).events_analytics(start_date=s_dt, end_date=e_dt)

    def processing(self) -> dict:
        with self.db_manager.session() as db:
            return AnalyticsRepository(db).processing_analytics()

    def timeline(self, *, start_date: str | None = None, end_date: str | None = None) -> dict:
        s_dt, e_dt = self._parse_date_range(start_date, end_date)
        with self.db_manager.session() as db:
            items = AnalyticsRepository(db).timeline(start_date=s_dt, end_date=e_dt)
            return {"items": items, "count": len(items)}
