import json
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_registry_preserves_sol_techniques_and_modes() -> None:
    registry = TechniqueRegistry.load(ROOT / "techniques" / "registry.toml")
    assert set(registry.names()) == {
        "kernel",
        "cache",
        "pisa",
        "topology",
        "quantization",
        "token_pruning",
    }
    assert registry["kernel"].correctness == "lossless"
    assert registry["topology"].correctness == "lossless"
    for name in ("cache", "pisa", "quantization", "token_pruning"):
        assert registry[name].correctness == "quality_gated"


def test_round_budgets_match_reviewed_contract() -> None:
    registry = TechniqueRegistry.load(ROOT / "techniques" / "registry.toml")
    assert registry["kernel"].round_budget == 40
    assert registry["cache"].round_budget == 20
    assert registry["pisa"].round_budget == 20
    assert registry["topology"].round_budget == 20
    assert registry["quantization"].round_budget == 20
    assert registry["token_pruning"].round_budget == 20
    assert registry["quantization"].origin == "sol-engine-full-adaptation"
    assert registry["token_pruning"].origin == "sol-engine-full-adaptation"


def test_registry_default_pass_excludes_optional_topology() -> None:
    registry = TechniqueRegistry.load(ROOT / "techniques" / "registry.toml")
    assert registry.default_order == [
        "kernel",
        "cache",
        "pisa",
        "quantization",
        "token_pruning",
    ]
    assert registry["topology"].optional is True


def test_contract_preserves_correctness_split() -> None:
    contract = (ROOT / "contracts" / "sol_engine" / "loop-and-gate.md").read_text(
        encoding="utf-8"
    )
    assert "never rejected using output differences" in contract
    assert "LPIPS" in contract
    assert "multimodal" in contract
    assert "engagement" in contract


def test_source_lock_records_reviewed_sol_engine_revision() -> None:
    source_lock = json.loads(
        (ROOT / "contracts" / "sol_engine" / "source-lock.json").read_text(
            encoding="utf-8"
        )
    )
    commit = source_lock["commit"]
    assert commit == "cee25847afdd34bc656abcca126262200b088dc8"
    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)
    assert source_lock["authoritative_paths"] == [
        "orchestration/prompts/loop_and_gate_contract.md",
        "orchestration/prompts/master.md",
        "orchestration/techniques.toml",
        "workflow/kernel_aw/nodes/codex_executor/kernel_scope.md",
        "workflow/cache_ca/nodes/codex_executor/cache_scope.md",
        "workflow/attention_pa/nodes/codex_executor/attention_scope.md",
        "workflow/topology_ta/nodes/codex_executor/topology_scope.md",
        "tools/vision/lpips_judge.py",
    ]
    source_hashes = json.loads(
        (ROOT / "contracts" / "sol_engine" / "source-hashes.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_hashes["commit"] == commit
    assert set(source_hashes["hashes"]) == set(source_lock["authoritative_paths"])
    assert all(len(digest) == 64 for digest in source_hashes["hashes"].values())


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
