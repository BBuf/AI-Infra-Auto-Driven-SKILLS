from __future__ import annotations

import json
from pathlib import Path

from .models import (
    AgentProfile,
    BaselineRecord,
    CampaignGoal,
    CandidateManifest,
    Delivery,
    EngagementReceipt,
    IntegratedDelivery,
    ProfileDigest,
    SourceLock,
    TechniqueContract,
    UnreachableCertificate,
)
from .request import LaunchRequest


SCHEMA_MODELS = {
    "goal": CampaignGoal,
    "baseline": BaselineRecord,
    "source-lock": SourceLock,
    "technique": TechniqueContract,
    "candidate": CandidateManifest,
    "delivery": Delivery,
    "integrated-delivery": IntegratedDelivery,
    "agent-profile": AgentProfile,
    "engagement-receipt": EngagementReceipt,
    "profile-digest": ProfileDigest,
    "unreachable-certificate": UnreachableCertificate,
    "launch-request": LaunchRequest,
}


def _stable_schema(value: object) -> object:
    """Normalize non-semantic Pydantic JSON Schema version differences."""
    if isinstance(value, list):
        return [_stable_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _stable_schema(item)
        for key, item in value.items()
        if not (key == "additionalProperties" and item is True)
    }
    if "const" in normalized:
        normalized.pop("type", None)
    if normalized.get("type") == "number":
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            item = normalized.get(key)
            if isinstance(item, int) and not isinstance(item, bool):
                normalized[key] = float(item)
    return normalized


def write_schemas(output_dir: Path) -> None:
    """Write deterministic JSON schemas for every public artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        target = output_dir / f"{name}.schema.json"
        target.write_text(
            json.dumps(
                _stable_schema(model.model_json_schema()),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
