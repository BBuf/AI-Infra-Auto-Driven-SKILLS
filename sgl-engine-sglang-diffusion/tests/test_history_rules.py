from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.history_rules import HistoryRuleCatalog
from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry


ROOT = Path(__file__).resolve().parents[1]


def registry() -> TechniqueRegistry:
    return TechniqueRegistry.load(ROOT / "techniques/registry.toml")


def test_checked_in_history_rules_are_complete_and_lane_scoped() -> None:
    catalog = HistoryRuleCatalog.load(ROOT / "knowledge/history-rules.toml", registry())
    assert len(catalog.sha256) == 64
    assert {rule.technique for rule in catalog.rules} == {
        "residency",
        "kernel",
        "cache",
        "quantization",
        "token_pruning",
    }
    residency = catalog.for_technique("residency")
    assert {rule.id for rule in residency} == {
        "residency.component-offload-removal",
        "residency.partial-dit-layers",
        "residency.memory-aware-load-order",
    }
    rendered = catalog.render("residency")
    assert catalog.sha256 in rendered
    assert "residency.partial-dit-layers" in rendered
    assert "kernel.regional-compile-graph" not in rendered


def test_every_catalog_source_is_present_in_manual_diff_audit() -> None:
    catalog = HistoryRuleCatalog.load(ROOT / "knowledge/history-rules.toml", registry())
    dossier = (ROOT.parent / "docs/references/sglang-diffusion-pr-rule-audit.md").read_text(
        encoding="utf-8"
    )
    for rule in catalog.rules:
        for source in rule.sources:
            assert source.pr_url in dossier
            assert source.merge_commit in dossier
            assert f"PR #{source.pr_url.rsplit('/', 1)[1]}" in dossier


def write_catalog(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "rules.toml"
    path.write_text(body, encoding="utf-8")
    return path


def valid_rule(*, rule_id: str = "residency.example", technique: str = "residency") -> str:
    return f'''schema_version = 1
[[rules]]
id = "{rule_id}"
technique = "{technique}"
correctness = "lossless"
summary = "measured example"
triggers = ["profile signal"]
actions = ["measured action"]
evidence = ["full run"]
incompatibilities = []
[[rules.sources]]
pr_url = "https://github.com/sgl-project/sglang/pull/21248"
merge_commit = "e4ad10520b8d409c6d32079a9c46ec7bdc0463ed"
validation = "reviewed"
'''


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            valid_rule() + valid_rule(),
            "duplicate history rule ID",
        ),
        (valid_rule(technique="missing"), "unknown technique"),
        (valid_rule().replace('correctness = "lossless"', 'correctness = "quality_gated"'), "correctness drifts"),
        (valid_rule().replace("https://github.com/sgl-project/sglang/pull/21248", "https://example.com/21248"), "malformed PR URL"),
        (valid_rule().replace("e4ad10520b8d409c6d32079a9c46ec7bdc0463ed", "deadbeef"), "malformed merge commit"),
        (valid_rule().replace('triggers = ["profile signal"]', "triggers = []"), "requires nonempty triggers"),
    ],
)
def test_history_rule_catalog_rejects_invalid_entries(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HistoryRuleCatalog.load(write_catalog(tmp_path, body), registry())
