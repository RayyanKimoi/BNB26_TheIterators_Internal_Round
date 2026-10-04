"""Slack webhook alert, fired when a diagnosis names a real failure class.

Uses the stdlib (`urllib.request`), no new dependency: the payload is one
small JSON POST. Every failure mode here is swallowed and logged, never
raised — a Slack outage, a bad webhook URL, or no `SLACK_WEBHOOK_URL` at all
must never break `POST /runs/{id}/diagnose`, which is the one thing this
project cannot afford to let an optional notification take down.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("backend.alerts")

# Overridable in tests; the real app always reads the live environment.
REQUEST_TIMEOUT_S = 5.0


def _dashboard_url(run_id: str) -> str:
    base = os.environ.get("DASHBOARD_BASE_URL", "http://localhost:5173").rstrip("/")
    return f"{base}/trace/{run_id}"


def build_slack_payload(
    run_id: str,
    task_type: str,
    flagged_step_index: int | None,
    predicted_class: str,
    evidence_summary: str,
) -> dict[str, Any]:
    """Slack Block Kit payload. Pure function, so the format is testable
    without a network call or a real webhook."""
    link = _dashboard_url(run_id)
    return {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Black Box flagged {predicted_class}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Run ID*\n`{run_id}`"},
                    {"type": "mrkdwn", "text": f"*Task Type*\n{task_type}"},
                    {"type": "mrkdwn", "text": f"*Flagged Step*\n{flagged_step_index}"},
                    {"type": "mrkdwn", "text": f"*Predicted Class*\n{predicted_class}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Evidence*\n{evidence_summary}"},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Open trace"},
                        "url": link,
                    }
                ],
            },
        ]
    }


def send_diagnosis_alert(
    run_id: str,
    task_type: str,
    flagged_step_index: int | None,
    predicted_class: str,
    evidence: dict[str, Any] | None,
) -> bool:
    """Best-effort Slack notification. Returns whether it actually sent,
    purely for tests and logging — callers in the request path ignore it.

    Never fires for `predicted_class == "unknown"`: that is the invariant
    tier's honest "I don't know", not a finding worth paging someone about.
    """
    if predicted_class == "unknown":
        return False

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return False

    evidence_summary = ", ".join(
        f"{k}={v}" for k, v in (evidence or {}).items() if v is not None
    ) or "no evidence recorded"

    payload = build_slack_payload(
        run_id, task_type, flagged_step_index, predicted_class, evidence_summary
    )

    try:
        request = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S)
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Non-blocking by design: a Slack outage must never fail a diagnosis.
        logger.warning("Slack alert failed for run %s: %s", run_id, exc)
        return False
