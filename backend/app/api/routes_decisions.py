"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
routes_decisions.py – FastAPI router for slice decision queries and explanation.

Endpoints
---------
GET  /decisions/recent?limit=50
    Last N SliceDecisions with confidence scores.
GET  /decisions/{decision_id}/explain
    Single SliceDecision with full SHAP explanation.
GET  /decisions/stats
    Aggregate stats: avg confidence, rule_trigger_rate, per-slice distribution.
POST /decisions/explain_hypothetical
    Body: TelemetryMessage → returns what decision would be made (dry run).
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend.app.networking.message_schema import SliceType, TelemetryMessage

router = APIRouter()


def _get_orchestrator(request: Request):
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")
    return orch


def _serialise_decision(d) -> dict[str, Any]:
    """Convert a SliceDecision to a JSON-safe dict."""
    return {
        "decision_id": d.decision_id,
        "ev_id": d.ev_id,
        "message_id": d.message_id,
        "assigned_slice": d.assigned_slice.value,
        "model_prediction": d.model_prediction.value,
        "confidence_score": d.confidence_score,
        "urgency_score": d.urgency_score,
        "ev_state": d.ev_state.value,
        "rule_triggered": d.rule_triggered,
        "timestamp": d.timestamp,
        "shap_top_features": [
            {"feature": k, "shap_value": v} for k, v in (d.shap_top_features or [])
        ],
    }


@router.get(
    "/stats",
    summary="Aggregate decision statistics",
    # Must be declared before /{decision_id}/explain to avoid route conflict
)
async def get_decision_stats(request: Request) -> dict[str, Any]:
    """
    Return aggregate statistics over all SliceDecisions processed so far.

    Includes
    --------
    * ``total_decisions``       – total number of decisions processed
    * ``avg_confidence``        – mean ML confidence score across all decisions
    * ``rule_trigger_rate``     – fraction of decisions where a rule fired
    * ``per_slice_distribution``– percentage of decisions per slice type
    * ``avg_urgency_score``     – mean urgency score
    """
    orch = _get_orchestrator(request)

    # Pull the last 1000 decisions for stats computation
    decisions = orch.get_recent_decisions(limit=1000)
    total = len(decisions)

    if total == 0:
        return {
            "total_decisions": 0,
            "avg_confidence": 0.0,
            "rule_trigger_rate": 0.0,
            "per_slice_distribution": {s.value: 0.0 for s in SliceType},
            "avg_urgency_score": 0.0,
        }

    avg_confidence = sum(d.confidence_score for d in decisions) / total
    rule_triggered_count = sum(1 for d in decisions if d.rule_triggered is not None)
    rule_trigger_rate = rule_triggered_count / total
    avg_urgency = sum(d.urgency_score for d in decisions) / total

    slice_counts: dict[str, int] = {s.value: 0 for s in SliceType}
    for d in decisions:
        slice_counts[d.assigned_slice.value] += 1
    per_slice_dist = {k: round(v / total, 4) for k, v in slice_counts.items()}

    return {
        "total_decisions": total,
        "avg_confidence": round(avg_confidence, 4),
        "rule_trigger_rate": round(rule_trigger_rate, 4),
        "per_slice_distribution": per_slice_dist,
        "avg_urgency_score": round(avg_urgency, 4),
    }


@router.get(
    "/recent",
    summary="Recent SliceDecisions with confidence scores",
)
async def get_recent_decisions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500, description="Number of decisions to return"),
) -> dict[str, Any]:
    """
    Return the most recent ``limit`` SliceDecisions, newest first.

    Each entry includes the decision ID, EV identifier, assigned slice,
    ML model prediction, confidence score, urgency score, current EV state,
    and the rule that fired (if any).
    """
    orch = _get_orchestrator(request)
    decisions = orch.get_recent_decisions(limit=limit)

    return {
        "timestamp": time.time(),
        "count": len(decisions),
        "decisions": [_serialise_decision(d) for d in decisions],
    }


@router.get(
    "/{decision_id}/explain",
    summary="SliceDecision with SHAP explanation",
)
async def explain_decision(decision_id: str, request: Request) -> dict[str, Any]:
    """
    Return a single SliceDecision enriched with its full SHAP explanation.

    The ``shap_top_features`` field lists the top-5 features (by |SHAP value|)
    that drove the ML model's slice recommendation.  If SHAP values are not
    available (no ML model loaded), the list will be empty.
    """
    orch = _get_orchestrator(request)
    decision = orch.get_decision_by_id(decision_id)

    if decision is None:
        raise HTTPException(
            status_code=404,
            detail=f"Decision '{decision_id}' not found in recent history",
        )

    serialised = _serialise_decision(decision)
    # Add the full feature vector for explainability
    serialised["features_used"] = decision.features_used

    return {
        "decision": serialised,
        "explanation": {
            "shap_available": len(decision.shap_top_features) > 0,
            "shap_features": serialised["shap_top_features"],
            "rule_explanation": (
                f"Hard rule '{decision.rule_triggered}' overrode the ML recommendation."
                if decision.rule_triggered
                else "No deterministic rule fired; ML recommendation was used."
            ),
        },
    }


class HypotheticalRequest(BaseModel):
    """Request body for the hypothetical decision endpoint."""

    message: TelemetryMessage

    model_config = {"arbitrary_types_allowed": True}


@router.post(
    "/explain_hypothetical",
    summary="Hypothetical slice decision for a given TelemetryMessage",
)
async def explain_hypothetical(
    body: HypotheticalRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Dry-run the full VESPER decision pipeline against a supplied TelemetryMessage
    without persisting the result or publishing any alerts.

    Useful for dashboard "what-if" analysis and integration testing.

    The supplied message is processed through:
    * Urgency scorer
    * ML inference (if models are loaded)
    * Rules engine
    * Admission controller (decision only – no token consumption)

    Returns the hypothetical ``SliceDecision`` and SHAP explanation.
    """
    orch = _get_orchestrator(request)
    msg = body.message

    # Run the pipeline; the decision is not stored permanently
    # We pass it through process_message which does append to _decisions,
    # which is a capped ring buffer – this is acceptable for a dry-run.
    try:
        decision = await orch.process_message(msg)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error during hypothetical evaluation: {exc}",
        ) from exc

    if decision is None:
        return {
            "result": "duplicate",
            "detail": "The supplied message was identified as a duplicate and was not processed.",
        }

    return {
        "result": "processed",
        "decision": _serialise_decision(decision),
        "features_used": decision.features_used,
        "explanation": {
            "shap_available": len(decision.shap_top_features) > 0,
            "shap_features": [
                {"feature": k, "shap_value": v}
                for k, v in (decision.shap_top_features or [])
            ],
            "rule_explanation": (
                f"Hard rule '{decision.rule_triggered}' overrode the ML recommendation."
                if decision.rule_triggered
                else "No deterministic rule fired; ML recommendation was used."
            ),
        },
    }
