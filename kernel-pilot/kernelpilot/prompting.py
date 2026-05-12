from __future__ import annotations

from .config import KernelTask, TaskSet
from .knowledge import load_catalog, render_catalog_for_prompt
from .lineage import LineageEntry


def build_iteration_prompt(
    taskset: TaskSet,
    task: KernelTask,
    lineage: list[LineageEntry],
) -> str:
    best = max(
        lineage,
        key=lambda entry: float(entry.score.get("geomean_vs_baseline", 0.0)),
        default=None,
    )
    best_text = "No committed KernelPilot candidate exists yet."
    if best is not None:
        best_text = (
            f"Best committed candidate: {best.version}, "
            f"geomean_vs_baseline={best.score.get('geomean_vs_baseline', 0.0)}."
        )

    rules = "\n".join(f"- {rule}" for rule in taskset.global_rules)
    forbidden = "\n".join(f"- {path}" for path in task.forbidden_files)
    correctness = "\n".join(f"- {item}" for item in task.correctness)
    search = "\n".join(f"- {item}" for item in task.search_space)
    baselines = "\n".join(
        f"- {row['case_id']}: baseline latency {row['baseline_latency_us']} us"
        for row in task.baseline
    )
    knowledge = render_catalog_for_prompt(load_catalog(), lane=task.lane)

    return f"""You are the KernelPilot iteration agent for one kernel task.

Task: {task.task_id}
Title: {task.title}
Base repository: {taskset.base_repo} at {taskset.base_ref}
Method reference: {taskset.paper_url}
Baseline source: {task.baseline_url}
Implementation lane: {task.lane}

KernelPilot operating rules:
{rules}

Forbidden implementation inputs:
{forbidden}

Semantic spec:
{task.semantic_spec}

Correctness gates:
{correctness}

Benchmark command:
{task.benchmark_command}

Baseline cases to beat:
{baselines}

Search-space hints:
{search}

Knowledge base K to consult before edits:
{knowledge}

Current lineage:
{best_text}

Iteration objective:
1. Start from the semantic spec and write a fresh candidate implementation.
2. Run correctness first, then benchmark every configured case.
3. If correctness fails or speed regresses, diagnose and revise before commit.
4. Commit only a candidate that is correct and improves the best lineage score.
5. Record failed directions in notes so the supervisor can redirect later.
"""
