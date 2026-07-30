from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sgl_engine_sglang_diffusion.artifacts import SCHEMA_MODELS, write_schemas
from sgl_engine_sglang_diffusion.models import Delivery


EXPECTED_SCHEMAS = {
    "agent-profile.schema.json",
    "baseline.schema.json",
    "candidate.schema.json",
    "delivery.schema.json",
    "engagement-receipt.schema.json",
    "goal.schema.json",
    "integrated-delivery.schema.json",
    "launch-request.schema.json",
    "profile-digest.schema.json",
    "source-lock.schema.json",
    "technique.schema.json",
    "unreachable-certificate.schema.json",
}


def test_delivery_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Delivery.model_validate(
            {
                "schema_version": 2,
                "status": "complete",
                "component": "kernel",
                "model_id": "test/model",
                "baseline": {},
                "frontier_points": [],
                "pareto_assessment": "empty",
                "fabricated": True,
            }
        )


def test_write_schemas_emits_exact_public_artifact_set(tmp_path: Path) -> None:
    write_schemas(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_SCHEMAS
    assert {f"{name}.schema.json" for name in SCHEMA_MODELS} == EXPECTED_SCHEMAS


def test_checked_in_schemas_match_generated_schemas(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    checked_in = package_root / "schemas"
    write_schemas(tmp_path)

    assert {path.name for path in checked_in.iterdir()} == EXPECTED_SCHEMAS
    for name in EXPECTED_SCHEMAS:
        assert (checked_in / name).read_bytes() == (tmp_path / name).read_bytes()
