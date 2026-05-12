from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LineageEntry:
    version: str
    task_id: str
    candidate_ref: str
    score: dict[str, Any]
    notes: str


class LineageStore:
    """Persistent single-lineage store used by the KernelPilot loop."""

    def __init__(self, root: str | Path, task_id: str) -> None:
        self.root = Path(root)
        self.task_id = task_id
        self.path = self.root / task_id / "lineage.json"

    def load(self) -> list[LineageEntry]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [LineageEntry(**entry) for entry in raw.get("entries", [])]

    def append(self, entry: LineageEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.load()
        entries.append(entry)
        payload = {"task_id": self.task_id, "entries": [asdict(item) for item in entries]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def best(self) -> LineageEntry | None:
        entries = self.load()
        if not entries:
            return None
        return max(
            entries,
            key=lambda item: float(item.score.get("geomean_vs_baseline", 0.0)),
        )
