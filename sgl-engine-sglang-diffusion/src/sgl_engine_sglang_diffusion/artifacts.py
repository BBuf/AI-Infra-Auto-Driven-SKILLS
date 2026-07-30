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
}


def write_schemas(output_dir: Path) -> None:
    """Write deterministic JSON schemas for every public artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        target = output_dir / f"{name}.schema.json"
        target.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
