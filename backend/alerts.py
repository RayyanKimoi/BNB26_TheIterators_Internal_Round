"""Slack webhook alert, fired when a diagnosis names a real failure class.

Uses `httpx`, already a dependency via FastAPI's test client, so this adds
nothing to the install. Every failure mode here is swallowed and logged,
never raised — a Slack outage, a bad webhook URL, or no `SLACK_WEBHOOK_URL`
at all must never break `POST /runs/{id}/diagnose`, which is the one thing
this project cannot afford to let an optional notification take down.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

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
    confidence: float | None = None,
) -> dict[str, Any]:
    """Slack Block Kit payload. Pure function, so the format is testable
    without a network call or a real webhook.

    No emoji in the header: CLAUDE.md bans emoji icons project-wide, and the
    alert is part of the product's voice even though it renders in Slack.
    """
    link = _dashboard_url(run_id)
    confidence_text = "not recorded" if confidence is None else f"{confidence:.0%}"
    return {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Agent failure flagged: {predicted_class}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Run ID*\n`{run_id}`"},
                    {"type": "mrkdwn", "text": f"*Task Type*\n{task_type}"},
                    {"type": "mrkdwn", "text": f"*Flagged Step*\n{flagged_step_index}"},
                    {"type": "mrkdwn", "text": f"*Predicted Class*\n{predicted_class}"},
                    {"type": "mrkdwn", "text": f"*Confidence*\n{confidence_text}"},
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
    confidence: float | None = None,
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
        run_id, task_type, flagged_step_index, predicted_class, evidence_summary, confidence
    )

    try:
        response = httpx.post(webhook_url, json=payload, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        return True
    except (httpx.HTTPError, OSError, ValueError) as exc:
        # Non-blocking by design: a Slack outage must never fail a diagnosis.
        logger.warning("Slack alert failed for run %s: %s", run_id, exc)
        return False
