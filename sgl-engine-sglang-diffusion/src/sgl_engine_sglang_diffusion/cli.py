from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import load_goal
from .models import CampaignGoal
from .state import StateStore
from .watchdog import CampaignWatchdog


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _campaign_store(campaign: Path) -> StateStore:
    return StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")


def _load_manifest(campaign: Path) -> dict[str, Any]:
    path = campaign / "CAMPAIGN.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CAMPAIGN.json must contain an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def initialize_goal(goal: CampaignGoal, run_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    identifier = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    campaign = run_root.resolve() / identifier
    campaign.mkdir(parents=True, exist_ok=False)

    prompt_target = campaign / "validation-prompts.txt"
    shutil.copy2(goal.workload.prompts, prompt_target)
    goal_payload = goal.model_dump(mode="json")
    goal_payload["workload"]["prompts"] = prompt_target.name
    frozen_goal = campaign / "GOAL.yaml"
    frozen_goal.write_text(
        yaml.safe_dump(goal_payload, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "campaign_id": identifier,
        "execution_mode": "interactive_single_agent",
        "created_at": datetime.now(UTC).isoformat(),
        "goal_sha256": hashlib.sha256(frozen_goal.read_bytes()).hexdigest(),
        "controller_command": [
            sys.executable,
            "-m",
            "sgl_engine_sglang_diffusion.cli",
            "resume",
            "--campaign",
            str(campaign),
        ],
    }
    _atomic_json(campaign / "CAMPAIGN.json", manifest)
    with _campaign_store(campaign) as store:
        store.create_campaign(identifier)
    return campaign


def initialize(goal_path: Path, run_root: Path) -> Path:
    return initialize_goal(load_goal(goal_path.resolve()), run_root)


def status_payload(campaign: Path) -> dict[str, Any]:
    campaign = campaign.resolve()
    manifest = _load_manifest(campaign)
    with _campaign_store(campaign) as store:
        return {
            "campaign_id": manifest["campaign_id"],
            "execution_mode": "interactive_single_agent",
            "status": store.status(manifest["campaign_id"]).value,
            "epoch": store.epoch(manifest["campaign_id"]),
            "campaign": str(campaign),
            "artifacts": sorted(
                path.name
                for path in campaign.iterdir()
                if path.is_file() and path.suffix in {".json", ".patch"}
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sgl-diffusion-engine",
        description="Persistent Sol-Engine-compatible SGLang Diffusion optimizer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--goal", type=Path, required=True)
    init.add_argument("--run-root", type=Path, required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--request", type=Path, required=True)
    launch.add_argument("--detach", action="store_true")

    work = subparsers.add_parser("work")
    work.add_argument("--campaign", type=Path, required=True)
    work.add_argument("--json", action="store_true")

    claim = subparsers.add_parser("claim")
    claim.add_argument("--campaign", type=Path, required=True)
    claim.add_argument("--technique", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--campaign", type=Path, required=True)
    submit.add_argument("--delivery", type=Path, required=True)

    skip = subparsers.add_parser("skip")
    skip.add_argument("--campaign", type=Path, required=True)
    skip.add_argument("--technique", required=True)
    skip.add_argument(
        "--classification",
        choices=("unsupported", "no_gain", "blocked"),
        required=True,
    )
    skip.add_argument("--reason", required=True)

    for name in (
        "run",
        "resume",
        "status",
        "progress",
        "sync-knowledge",
        "package",
        "watchdog",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name in {"status", "progress"}:
            command.add_argument("--json", action="store_true")
        if name == "progress":
            command.add_argument("--watch", action="store_true")
            command.add_argument("--interval", type=float, default=5.0)
        if name == "watchdog":
            command.add_argument("--once", action="store_true")

    contracts = subparsers.add_parser("check-contracts")
    contracts.add_argument("--sol-checkout", type=Path, required=True)
    contracts.add_argument(
        "--source-lock",
        type=Path,
        default=PACKAGE_ROOT / "contracts/sol_engine/source-lock.json",
    )
    contracts.add_argument(
        "--hashes",
        type=Path,
        default=PACKAGE_ROOT / "contracts/sol_engine/source-hashes.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        campaign = initialize(args.goal, args.run_root)
        payload = status_payload(campaign)
        payload["artifacts"].extend(["CAMPAIGN.json", "GOAL.yaml"])
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "launch":
        from .launcher import launch_campaign

        payload = launch_campaign(args.request.resolve(), detach=args.detach)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        payload = status_payload(args.campaign)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"{payload['campaign_id']}: {payload['status']} "
                f"(epoch {payload['epoch']})"
            )
        return 0
    if args.command == "progress":
        from .progress import watch_progress, write_progress

        if args.watch:
            watch_progress(
                args.campaign.resolve(),
                interval_seconds=args.interval,
                json_output=args.json,
            )
        else:
            projection = write_progress(args.campaign.resolve())
            if args.json:
                print(json.dumps(projection, indent=2, sort_keys=True))
            else:
                from .progress import render_progress

                print(render_progress(projection))
        return 0
    if args.command in {"work", "claim", "submit", "skip"}:
        from .work_orders import WorkOrderManager

        campaign = args.campaign.resolve()
        manifest = _load_manifest(campaign)
        campaign_id = str(manifest["campaign_id"])
        with _campaign_store(campaign) as store:
            manager = WorkOrderManager(
                campaign,
                campaign_id=campaign_id,
                store=store,
            )
            if args.command == "work":
                payload = manager.work()
            elif args.command == "claim":
                order = manager.claim(args.technique)
                payload = manager.work()
                payload["claimed_work_order"] = order.model_dump(mode="json")
            elif args.command == "skip":
                disposition = manager.skip(
                    args.technique,
                    classification=args.classification,
                    reason=args.reason,
                )
                payload = manager.work()
                payload["disposition"] = disposition.model_dump(mode="json")
            else:
                submission = manager.submit(args.delivery)
                payload = {
                    "campaign_id": campaign_id,
                    "execution_mode": "interactive_single_agent",
                    "status": store.status(campaign_id).value,
                    "submission": submission,
                }
        should_advance = args.command == "submit" or (
            args.command == "skip" and payload["status"] == "INTEGRATING"
        )
        if should_advance:
            from .runtime import run_campaign_command

            payload["verification"] = run_campaign_command("resume", campaign)
            with _campaign_store(campaign) as store:
                manager = WorkOrderManager(
                    campaign,
                    campaign_id=campaign_id,
                    store=store,
                )
                payload.update(manager.work())
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "watchdog":
        campaign = args.campaign.resolve()
        with _campaign_store(campaign) as store:
            watchdog = CampaignWatchdog(campaign, store)
            if args.once:
                pid = watchdog.tick()
                print(json.dumps({"restarted_pid": pid}))
            else:
                watchdog.run_forever()
        return 0
    if args.command == "sync-knowledge":
        from .knowledge import load_registry, sync_source

        campaign = args.campaign.resolve()
        registry = load_registry(PACKAGE_ROOT / "knowledge/registry.toml")
        locks = json.loads((campaign / "SOURCE-LOCKS.json").read_text())
        snapshots: dict[str, str] = {}
        for name, patterns in registry.items():
            lock = locks[name]
            checkout = campaign / "source-worktrees" / name
            output = campaign / "knowledge" / name / lock["commit"]
            sync_source(
                name=name,
                checkout=checkout,
                commit=lock["commit"],
                patterns=patterns,
                output_dir=output,
            )
            snapshots[name] = str(output / "index.json")
        print(json.dumps({"campaign": str(campaign), "snapshots": snapshots}))
        return 0
    if args.command == "check-contracts":
        from .knowledge import check_contract_hashes

        issues = check_contract_hashes(
            args.source_lock.resolve(),
            args.sol_checkout.resolve(),
            args.hashes.resolve(),
        )
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1 if issues else 0
    if args.command in {"run", "resume", "package"}:
        # The default runtime is imported lazily so status/init remain CPU-only.
        from .runtime import run_campaign_command

        result = run_campaign_command(args.command, args.campaign.resolve())
        from .progress import write_progress

        write_progress(args.campaign.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
