"""MongoDB Atlas persistence for uploaded-dataset health reports."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient


def save_report(
    mongo_uri: str,
    database_name: str,
    collection_name: str,
    analysis: Any,
) -> str:
    """Store a completed upload analysis without retaining the uploaded CSV itself."""
    document = asdict(analysis)
    document["saved_at"] = datetime.now(timezone.utc)
    document["report_type"] = "uploaded_dataset_health"

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        result = client[database_name][collection_name].insert_one(document)
        return str(result.inserted_id)
    finally:
        client.close()


def load_recent_reports(
    mongo_uri: str,
    database_name: str,
    collection_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return compact, presentation-friendly summaries of recently saved reports."""
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        cursor = (
            client[database_name][collection_name]
            .find({}, {"file_name": 1, "saved_at": 1, "data_health_score": 1, "health_status": 1, "row_count": 1, "column_count": 1})
            .sort("saved_at", -1)
            .limit(limit)
        )
        return [
            {
                "File Name": item.get("file_name", "Unknown"),
                "Saved At (UTC)": item.get("saved_at"),
                "Health Score": item.get("data_health_score"),
                "Status": item.get("health_status"),
                "Rows": item.get("row_count"),
                "Columns": item.get("column_count"),
            }
            for item in cursor
        ]
    finally:
        client.close()
