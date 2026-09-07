"""Final monitoring layer for behavioural drift and critical data-health events.

The core feature generation and statistical comparison remain in PySpark.
This module turns the compact fingerprint outputs into an idempotent alert
history that the dashboard can monitor.
"""

import json
from pathlib import Path
from datetime import datetime, timezone


BASE_DIR = Path(__file__).resolve().parent.parent
BEHAVIORAL_FILE = BASE_DIR / "fingerprints" / "behavioral_drift.json"
MONTHLY_FILE = BASE_DIR / "fingerprints" / "monthly_fingerprints.json"
ALERT_HISTORY_FILE = BASE_DIR / "fingerprints" / "alert_history.json"

CRITICAL_SIMILARITY_THRESHOLD = 35.0
ACTIVE_DATASET_YEAR = "2022"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def stable_alert_id(alert_type, batch):
    """Create one stable identity per alert type and source batch."""
    prefix = "behavioral" if alert_type == "Behavioral Drift" else "data-health"
    return f"{prefix}-{batch}"


def behavioral_alert(fingerprint):
    """Return an alert only for statistically established behavioural drift."""
    if not fingerprint.get("baseline_ready"):
        return None

    status = fingerprint["drift_status"]
    score = fingerprint["behavioral_similarity_score"]

    if status in {"Normal", "Baseline Building", "Insufficient History"}:
        return None

    if score < CRITICAL_SIMILARITY_THRESHOLD:
        severity = "CRITICAL"
    elif status == "Major Drift":
        severity = "HIGH"
    else:
        severity = "WARNING"

    feature_names = [
        item["feature"].replace("_", " ")
        for item in fingerprint["top_deviating_features"]
    ]
    trigger = (
        f"{status}: behavioural similarity is {score}/100. "
        f"Top deviations: {', '.join(feature_names) or 'not available'}."
    )

    if fingerprint.get("is_partial_source_window"):
        trigger += (
            " " + fingerprint["source_window_note"]
            + " Transaction volume is excluded from behavioural similarity "
              "scoring for this source-truncated week."
        )

    return {
        "alert_id": stable_alert_id("Behavioral Drift", fingerprint["week"]),
        "alert_type": "Behavioral Drift",
        "batch": fingerprint["week"],
        "batch_timestamp": fingerprint["week"],
        "severity": severity,
        "similarity_score": score,
        "drift_status": status,
        "trigger": trigger,
        "top_deviating_features": fingerprint["top_deviating_features"],
        "source_window_note": fingerprint.get("source_window_note"),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def technical_health_alert(fingerprint):
    """Escalate only Implementation 1's genuinely critical health outcomes."""
    if fingerprint["health_status"] != "Critical":
        return None

    return {
        "alert_id": stable_alert_id("Data Health", fingerprint["month"]),
        "alert_type": "Data Health",
        "batch": fingerprint["month"],
        "batch_timestamp": fingerprint["month"],
        "severity": "CRITICAL",
        "similarity_score": None,
        "drift_status": None,
        "trigger": (
            f"Implementation 1 Data Health Score is {fingerprint['health_score']}/100: "
            f"{'; '.join(fingerprint['data_quality_anomalies'])}"
        ),
        "top_deviating_features": fingerprint["data_quality_anomalies"],
        "source_window_note": (
            None
        ),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_alerts(behavioral_fingerprints, monthly_fingerprints):
    alerts = []

    for fingerprint in behavioral_fingerprints:
        alert = behavioral_alert(fingerprint)
        if alert:
            alerts.append(alert)

    for fingerprint in monthly_fingerprints:
        alert = technical_health_alert(fingerprint)
        if alert:
            alerts.append(alert)

    return sorted(alerts, key=lambda item: (item["batch_timestamp"], item["alert_id"]))


def merge_history(existing_alerts, new_alerts):
    """Merge by alert type and batch, including migration of legacy IDs."""
    alerts_by_identity = {
        (alert["alert_type"], alert["batch"]): alert
        for alert in existing_alerts
    }
    alerts_by_identity.update({
        (alert["alert_type"], alert["batch"]): alert
        for alert in new_alerts
    })
    return sorted(
        alerts_by_identity.values(),
        key=lambda item: (item["batch_timestamp"], item["alert_id"]),
    )


def main():
    if not BEHAVIORAL_FILE.exists():
        raise FileNotFoundError(
            "Behavioral fingerprints are missing. Run src/behavioral_drift.py first."
        )

    behavioral_fingerprints = load_json(BEHAVIORAL_FILE)
    monthly_fingerprints = load_json(MONTHLY_FILE)
    existing_alerts = load_json(ALERT_HISTORY_FILE) if ALERT_HISTORY_FILE.exists() else []
    # The active final pipeline is the Amazon 2022 dataset. Retain its prior
    # alert history across reruns, but never carry legacy alerts forward.
    existing_alerts = [
        alert for alert in existing_alerts
        if str(alert["batch"]).startswith(ACTIVE_DATASET_YEAR)
    ]

    new_alerts = build_alerts(behavioral_fingerprints, monthly_fingerprints)
    alert_history = merge_history(existing_alerts, new_alerts)

    with open(ALERT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(alert_history, file, indent=4)

    print(f"Generated or updated {len(new_alerts)} monitored alerts.")
    print(f"Alert history contains {len(alert_history)} unique alerts.")
    print(f"Saved to: {ALERT_HISTORY_FILE}")


if __name__ == "__main__":
    main()
