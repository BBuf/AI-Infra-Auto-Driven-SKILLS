"""Validated, PR-derived optimization hypotheses for executor lanes."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from .techniques import TechniqueRegistry


_RULE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]+$")
_PR_URL = re.compile(r"^https://github\.com/sgl-project/sglang/pull/[1-9][0-9]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class HistoryRuleSource:
    pr_url: str
    merge_commit: str
    validation: str


@dataclass(frozen=True)
class HistoryRule:
    id: str
    technique: str
    correctness: str
    summary: str
    triggers: tuple[str, ...]
    actions: tuple[str, ...]
    evidence: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    sources: tuple[HistoryRuleSource, ...]


class HistoryRuleCatalog:
    """Immutable checked-in history catalog bound to the technique registry."""

    def __init__(self, rules: tuple[HistoryRule, ...], *, sha256: str):
        self.rules = rules
        self.sha256 = sha256

    @classmethod
    def load(
        cls, path: Path, technique_registry: TechniqueRegistry
    ) -> HistoryRuleCatalog:
        raw_bytes = path.read_bytes()
        data = tomllib.loads(raw_bytes.decode("utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("unsupported history rule catalog schema_version")
        raw_rules = data.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("history rule catalog must define rules")

        rules: list[HistoryRule] = []
        seen: set[str] = set()
        known_techniques = set(technique_registry.names())
        for raw in raw_rules:
            if not isinstance(raw, dict):
                raise ValueError("history rule entry must be a table")
            rule_id = str(raw.get("id", ""))
            if not _RULE_ID.fullmatch(rule_id):
                raise ValueError(f"invalid history rule ID: {rule_id!r}")
            if rule_id in seen:
                raise ValueError(f"duplicate history rule ID: {rule_id}")
            seen.add(rule_id)

            technique = str(raw.get("technique", ""))
            if technique not in known_techniques:
                raise ValueError(
                    f"history rule {rule_id} has unknown technique: {technique!r}"
                )
            correctness = str(raw.get("correctness", ""))
            if correctness != technique_registry[technique].correctness:
                raise ValueError(
                    f"history rule {rule_id} correctness drifts from technique registry"
                )

            sources = cls._sources(rule_id, raw.get("sources"))
            rules.append(
                HistoryRule(
                    id=rule_id,
                    technique=technique,
                    correctness=correctness,
                    summary=cls._text(rule_id, raw, "summary"),
                    triggers=cls._strings(rule_id, raw, "triggers"),
                    actions=cls._strings(rule_id, raw, "actions"),
                    evidence=cls._strings(rule_id, raw, "evidence"),
                    incompatibilities=cls._strings(
                        rule_id, raw, "incompatibilities", allow_empty=True
                    ),
                    sources=sources,
                )
            )
        return cls(tuple(rules), sha256=hashlib.sha256(raw_bytes).hexdigest())

    @staticmethod
    def _text(rule_id: str, raw: dict[str, object], name: str) -> str:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"history rule {rule_id} requires nonempty {name}")
        return value.strip()

    @staticmethod
    def _strings(
        rule_id: str,
        raw: dict[str, object],
        name: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[str, ...]:
        value = raw.get(name, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"history rule {rule_id} has invalid {name}")
        if not value and not allow_empty:
            raise ValueError(f"history rule {rule_id} requires nonempty {name}")
        return tuple(item.strip() for item in value)

    @staticmethod
    def _sources(rule_id: str, value: object) -> tuple[HistoryRuleSource, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"history rule {rule_id} requires sources")
        sources: list[HistoryRuleSource] = []
        for raw in value:
            if not isinstance(raw, dict):
                raise ValueError(f"history rule {rule_id} has invalid source")
            pr_url = str(raw.get("pr_url", ""))
            merge_commit = str(raw.get("merge_commit", ""))
            validation = str(raw.get("validation", "")).strip()
            if not _PR_URL.fullmatch(pr_url):
                raise ValueError(f"history rule {rule_id} has malformed PR URL")
            if not _COMMIT.fullmatch(merge_commit):
                raise ValueError(f"history rule {rule_id} has malformed merge commit")
            if not validation:
                raise ValueError(f"history rule {rule_id} source lacks validation")
            sources.append(HistoryRuleSource(pr_url, merge_commit, validation))
        return tuple(sources)

    def for_technique(self, technique: str) -> tuple[HistoryRule, ...]:
        return tuple(rule for rule in self.rules if rule.technique == technique)

    def render(self, technique: str) -> str:
        """Render a stable machine-readable lane subset for an executor prompt."""
        rules = self.for_technique(technique)
        payload = {
            "catalog_sha256": self.sha256,
            "policy": (
                "These diff-reviewed rules generate hypotheses only. Acceptance "
                "requires the active profile, frozen workload, and lane evidence."
            ),
            "technique": technique,
            "rules": [asdict(rule) for rule in rules],
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
