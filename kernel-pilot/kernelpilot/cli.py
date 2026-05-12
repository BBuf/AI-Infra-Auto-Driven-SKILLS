from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import ConfigError, load_taskset
from .evaluator import load_result
from .knowledge import load_catalog, render_catalog_for_prompt
from .lineage import LineageEntry, LineageStore
from .prompting import build_iteration_prompt
from .scoring import score_against_baseline
from .supervisor import supervisor_hints


def _load_lineage(args: argparse.Namespace) -> list[LineageEntry]:
    if not getattr(args, "lineage_root", None):
        return []
    return LineageStore(args.lineage_root, args.task).load()


def cmd_validate(args: argparse.Namespace) -> int:
    taskset = load_taskset(args.taskset)
    print(
        json.dumps(
            {
                "name": taskset.name,
                "base_repo": taskset.base_repo,
                "base_ref": taskset.base_ref,
                "tasks": [task.task_id for task in taskset.tasks],
            },
            indent=2,
        )
    )
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    taskset = load_taskset(args.taskset)
    task = taskset.task_by_id(args.task)
    lineage = _load_lineage(args)
    print(build_iteration_prompt(taskset, task, lineage))
    for hint in supervisor_hints(task, lineage):
        print(f"- supervisor: {hint}", file=sys.stderr)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    taskset = load_taskset(args.taskset)
    task = taskset.task_by_id(args.task)
    result = load_result(args.result)
    summary = score_against_baseline(task, result)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0 if summary.correct else 2


def cmd_record(args: argparse.Namespace) -> int:
    taskset = load_taskset(args.taskset)
    task = taskset.task_by_id(args.task)
    result = load_result(args.result)
    summary = score_against_baseline(task, result)
    store = LineageStore(args.lineage_root, args.task)
    entry = LineageEntry(
        version=args.version,
        task_id=task.task_id,
        candidate_ref=args.candidate_ref,
        score=asdict(summary),
        notes=Path(args.notes).read_text(encoding="utf-8") if args.notes else "",
    )
    store.append(entry)
    print(json.dumps({"recorded": args.version, "lineage": str(store.path)}, indent=2))
    return 0


def cmd_refs(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    print(render_catalog_for_prompt(catalog, lane=args.lane))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KernelPilot kernel iteration helper")
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--taskset", required=True)
    validate.set_defaults(func=cmd_validate)

    prompt = sub.add_parser("prompt")
    prompt.add_argument("--taskset", required=True)
    prompt.add_argument("--task", required=True)
    prompt.add_argument("--lineage-root")
    prompt.set_defaults(func=cmd_prompt)

    compare = sub.add_parser("compare")
    compare.add_argument("--taskset", required=True)
    compare.add_argument("--task", required=True)
    compare.add_argument("--result", required=True)
    compare.set_defaults(func=cmd_compare)

    record = sub.add_parser("record")
    record.add_argument("--taskset", required=True)
    record.add_argument("--task", required=True)
    record.add_argument("--result", required=True)
    record.add_argument("--lineage-root", required=True)
    record.add_argument("--version", required=True)
    record.add_argument("--candidate-ref", required=True)
    record.add_argument("--notes")
    record.set_defaults(func=cmd_record)

    refs = sub.add_parser("refs")
    refs.add_argument("--lane")
    refs.add_argument("--catalog")
    refs.set_defaults(func=cmd_refs)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
