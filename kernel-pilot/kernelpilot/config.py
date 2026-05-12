from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a KernelPilot taskset is malformed."""


@dataclass(frozen=True)
class KernelTask:
    task_id: str
    title: str
    baseline_id: str
    baseline_url: str
    operator: str
    lane: str
    semantic_spec: str
    forbidden_files: tuple[str, ...]
    correctness: tuple[str, ...]
    benchmark_command: str
    baseline: tuple[dict[str, Any], ...]
    search_space: tuple[str, ...]
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KernelTask":
        required = [
            "id",
            "title",
            "baseline_id",
            "baseline_url",
            "operator",
            "lane",
            "semantic_spec",
            "forbidden_files",
            "correctness",
            "benchmark_command",
            "baseline",
            "search_space",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise ConfigError(f"task missing required keys: {missing}")
        baseline = tuple(raw["baseline"])
        if not baseline:
            raise ConfigError(f"task {raw['id']} has empty baseline")
        for row in baseline:
            if "case_id" not in row or "baseline_latency_us" not in row:
                raise ConfigError(
                    f"task {raw['id']} baseline rows need case_id and baseline_latency_us"
                )
        return cls(
            task_id=str(raw["id"]),
            title=str(raw["title"]),
            baseline_id=str(raw["baseline_id"]),
            baseline_url=str(raw["baseline_url"]),
            operator=str(raw["operator"]),
            lane=str(raw["lane"]),
            semantic_spec=str(raw["semantic_spec"]),
            forbidden_files=tuple(str(x) for x in raw["forbidden_files"]),
            correctness=tuple(str(x) for x in raw["correctness"]),
            benchmark_command=str(raw["benchmark_command"]),
            baseline=baseline,
            search_space=tuple(str(x) for x in raw["search_space"]),
            notes=str(raw.get("notes", "")),
        )


@dataclass(frozen=True)
class TaskSet:
    name: str
    base_repo: str
    base_ref: str
    paper_url: str
    tasks: tuple[KernelTask, ...]
    global_rules: tuple[str, ...]

    def task_by_id(self, task_id: str) -> KernelTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        known = ", ".join(task.task_id for task in self.tasks)
        raise ConfigError(f"unknown task {task_id!r}; known tasks: {known}")


def load_taskset(path: str | Path) -> TaskSet:
    taskset_path = Path(path)
    raw = json.loads(taskset_path.read_text(encoding="utf-8"))
    required = ["name", "base_repo", "base_ref", "paper_url", "tasks", "global_rules"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ConfigError(f"taskset missing required keys: {missing}")
    tasks = tuple(KernelTask.from_dict(task) for task in raw["tasks"])
    seen: set[str] = set()
    duplicates: list[str] = []
    for task in tasks:
        if task.task_id in seen:
            duplicates.append(task.task_id)
        seen.add(task.task_id)
    if duplicates:
        raise ConfigError(f"duplicate task ids: {duplicates}")
    return TaskSet(
        name=str(raw["name"]),
        base_repo=str(raw["base_repo"]),
        base_ref=str(raw["base_ref"]),
        paper_url=str(raw["paper_url"]),
        tasks=tasks,
        global_rules=tuple(str(rule) for rule in raw["global_rules"]),
    )
