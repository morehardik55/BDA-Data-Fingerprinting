"""MongoDB Atlas persistence for uploaded dataset health reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


DEFAULT_DATABASE = "bda_monitoring"
DEFAULT_COLLECTION = "dataset_health_reports"


def _require_mongo_client() -> str | None:
    if MongoClient is None:
        return "MongoDB support is unavailable. Add pymongo to the deployment requirements."
    return None


def build_health_report(analysis) -> dict[str, Any]:
    """Create a JSON-compatible report without altering the original analysis."""
    return {
        "created_at": datetime.now(timezone.utc),
        "source": "uploaded_csv",
        "file_name": analysis.file_name,
        "row_count": analysis.row_count,
        "column_count": analysis.column_count,
        "data_health_score": analysis.data_health_score,
        "health_status": analysis.health_status,
        "overall_missing_percentage": analysis.overall_missing_percentage,
        "duplicate_percentage": analysis.duplicate_percentage,
        "detected_issues": analysis.detected_issues,
        "schema_overview": analysis.schema_overview,
        "missing_by_column": analysis.missing_by_column,
        "numerical_summary": analysis.numerical_summary,
        "outlier_analysis": analysis.outlier_analysis,
    }


def save_health_report(
    mongo_uri: str,
    analysis,
    database_name: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
) -> tuple[str | None, str | None]:
    """Store one uploaded-data analysis report in MongoDB Atlas."""
    dependency_error = _require_mongo_client()
    if dependency_error:
        return None, dependency_error

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        result = client[database_name][collection_name].insert_one(build_health_report(analysis))
        client.close()
        return str(result.inserted_id), None
    except Exception as exc:
        return None, f"Unable to save the report to MongoDB: {exc}"


def load_recent_health_reports(
    mongo_uri: str,
    database_name: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = 20,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Load recent stored reports for the dashboard history view."""
    dependency_error = _require_mongo_client()
    if dependency_error:
        return None, dependency_error

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        cursor = client[database_name][collection_name].find(
            {},
            {
                "_id": 0,
                "created_at": 1,
                "file_name": 1,
                "row_count": 1,
                "column_count": 1,
                "data_health_score": 1,
                "health_status": 1,
                "overall_missing_percentage": 1,
                "duplicate_percentage": 1,
            },
        ).sort("created_at", -1).limit(limit)
        reports = list(cursor)
        client.close()
        return reports, None
    except Exception as exc:
        return None, f"Unable to load MongoDB reports: {exc}"
