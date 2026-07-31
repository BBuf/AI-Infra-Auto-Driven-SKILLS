from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry
from sgl_engine_sglang_diffusion.resources import TECHNIQUE_REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def test_registry_preserves_techniques_and_modes() -> None:
    registry = TechniqueRegistry.load(TECHNIQUE_REGISTRY)
    assert set(registry.names()) == {
        "kernel",
        "cache",
        "sparse_attention",
        "topology",
        "quantization",
        "token_pruning",
    }
    assert registry["kernel"].correctness == "lossless"
    assert registry["topology"].correctness == "lossless"
    for name in ("cache", "sparse_attention", "quantization", "token_pruning"):
        assert registry[name].correctness == "quality_gated"


def test_round_budgets_match_reviewed_contract() -> None:
    registry = TechniqueRegistry.load(TECHNIQUE_REGISTRY)
    assert registry["kernel"].round_budget == 40
    assert registry["cache"].round_budget == 20
    assert registry["sparse_attention"].round_budget == 20
    assert registry["topology"].round_budget == 20
    assert registry["quantization"].round_budget == 20
    assert registry["token_pruning"].round_budget == 20
    assert all(
        registry[name].origin == "bundled-search-space" for name in registry.names()
    )


def test_registry_default_pass_excludes_optional_topology() -> None:
    registry = TechniqueRegistry.load(TECHNIQUE_REGISTRY)
    assert registry.default_order == [
        "kernel",
        "cache",
        "sparse_attention",
        "quantization",
        "token_pruning",
    ]
    assert registry["topology"].optional is True


def test_contract_preserves_correctness_split() -> None:
    contract = (ROOT / "contracts" / "verification.md").read_text(encoding="utf-8")
    assert "Never reject a lossless candidate using output differences" in contract
    assert "LPIPS" in contract
    assert "visual review" in contract
    assert "engagement" in contract


def test_registry_rejects_unknown_correctness_mode(tmp_path: Path) -> None:
    (tmp_path / "techniques").mkdir()
    scope = tmp_path / "techniques" / "example.md"
    scope.write_text("# Example\n", encoding="utf-8")
    registry = tmp_path / "techniques" / "registry.toml"
    registry.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'default_order = ["example"]',
                "[techniques.example]",
                'workflow_uid = "example"',
                'scope = "techniques/example.md"',
                'correctness = "numeric_tolerance"',
                "round_budget = 1",
                'origin = "test"',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid correctness mode"):
        TechniqueRegistry.load(registry)
