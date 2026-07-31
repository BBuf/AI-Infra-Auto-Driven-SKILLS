from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CORRECTNESS_MODES = frozenset({"lossless", "quality_gated"})


@dataclass(frozen=True)
class Technique:
    """A reviewed optimization lane and its immutable acceptance mode."""

    name: str
    workflow_uid: str
    scope: Path
    correctness: str
    round_budget: int
    origin: str
    optional: bool = False


class TechniqueRegistry:
    """Load and expose the bundled SGLang Diffusion technique registry."""

    def __init__(self, entries: dict[str, Technique], default_order: list[str]):
        self._entries = entries
        self.default_order = default_order

    @classmethod
    def load(cls, path: Path) -> TechniqueRegistry:
        path = path.resolve()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("unsupported technique registry schema_version")

        raw_techniques = data.get("techniques")
        if not isinstance(raw_techniques, dict) or not raw_techniques:
            raise ValueError("technique registry must define techniques")

        entries: dict[str, Technique] = {}
        for name, value in raw_techniques.items():
            if not isinstance(value, dict):
                raise ValueError(f"invalid technique entry: {name}")
            raw: dict[str, Any] = value
            scope = path.parent.parent / str(raw["scope"])
            if not scope.is_file():
                raise ValueError(f"missing technique scope: {scope}")

            correctness = str(raw["correctness"])
            if correctness not in _CORRECTNESS_MODES:
                raise ValueError(f"invalid correctness mode for {name}: {correctness}")
            round_budget = int(raw["round_budget"])
            if round_budget <= 0:
                raise ValueError(f"round_budget must be positive for {name}")

            entries[name] = Technique(
                name=name,
                workflow_uid=str(raw["workflow_uid"]),
                scope=scope,
                correctness=correctness,
                round_budget=round_budget,
                origin=str(raw["origin"]),
                optional=bool(raw.get("optional", False)),
            )

        raw_default_order = data.get("default_order")
        if not isinstance(raw_default_order, list):
            raise ValueError("default_order must be a list")
        default_order = [str(name) for name in raw_default_order]
        unknown_defaults = set(default_order) - entries.keys()
        if unknown_defaults:
            unknown = ", ".join(sorted(unknown_defaults))
            raise ValueError(f"default_order contains unknown techniques: {unknown}")
        if len(default_order) != len(set(default_order)):
            raise ValueError("default_order contains duplicate techniques")

        return cls(entries, default_order)

    def names(self) -> list[str]:
        return list(self._entries)

    def __getitem__(self, name: str) -> Technique:
        return self._entries[name]
