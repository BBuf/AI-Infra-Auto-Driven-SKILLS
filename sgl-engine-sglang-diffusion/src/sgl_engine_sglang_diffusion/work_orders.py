from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from .models import (
    AgentWorkOrder,
    CampaignStatus,
    CorrectnessMode,
    SourceLock,
    TechniqueDisposition,
)
from .resources import TECHNIQUE_REGISTRY
from .sources import SourceManager
from .state import StateStore
from .techniques import TechniqueRegistry


class WorkOrderError(RuntimeError):
    """The requested interactive action is unsafe in the current campaign."""


_CLOSED_CLASSIFICATIONS = frozenset({"unsupported", "no_gain"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkOrderError(f"invalid campaign artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise WorkOrderError(f"campaign artifact must contain an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class WorkOrderManager:
    """Own one serial work order for the current interactive root agent."""

    def __init__(
        self,
        campaign_dir: Path,
        *,
        campaign_id: str,
        store: StateStore,
        source_manager: SourceManager | None = None,
        registry: TechniqueRegistry | None = None,
    ) -> None:
        self.campaign_dir = campaign_dir.resolve()
        self.campaign_id = campaign_id
        self.store = store
        self.source_manager = source_manager or SourceManager(
            self.campaign_dir.parent / ".sgl-diffusion-source-cache"
        )
        self.registry = registry or TechniqueRegistry.load(TECHNIQUE_REGISTRY)

    @contextmanager
    def _campaign_lock(self) -> Iterator[None]:
        path = self.campaign_dir / ".work-order.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def claim(self, technique: str) -> AgentWorkOrder:
        with self._campaign_lock():
            status = self.store.status(self.campaign_id)
            if status is not CampaignStatus.AWAITING_AGENT:
                raise WorkOrderError(
                    "claim requires AWAITING_AGENT, current status is "
                    f"{status.value}"
                )
            if technique not in self._routes():
                raise WorkOrderError(
                    f"technique {technique!r} is not a routed suggestion"
                )
            disposition = self._dispositions().get(technique)
            if disposition is not None and disposition.closed:
                raise WorkOrderError(
                    f"technique {technique!r} is closed as "
                    f"{disposition.classification}"
                )
            entry = self.registry[technique]
            used = self._rounds_used(technique)
            if used >= entry.round_budget:
                raise WorkOrderError(
                    f"technique {technique!r} exhausted its scientific round budget"
                )

            epoch = self.store.epoch(self.campaign_id) + 1
            search_root = self.campaign_dir / "search" / str(epoch)
            work_order_path = search_root / "AGENT-WORK.json"
            worktree = search_root / "worktree"
            if work_order_path.exists() or worktree.exists():
                raise WorkOrderError(
                    f"predicted epoch {epoch} already has interactive work artifacts"
                )

            source_locks_path = self.campaign_dir / "SOURCE-LOCKS.json"
            baseline_path = self.campaign_dir / "BASELINE.json"
            profile_path = self.campaign_dir / "profiles" / "0" / "PROFILE-DIGEST.json"
            knowledge_manifest_path = self.campaign_dir / "KNOWLEDGE.json"
            search_space_path = self.campaign_dir / "SEARCH-SPACE.json"
            for required in (
                source_locks_path,
                baseline_path,
                profile_path,
                knowledge_manifest_path,
                search_space_path,
            ):
                if not required.is_file():
                    raise WorkOrderError(
                        f"cannot claim work before required artifact exists: {required}"
                    )
            source_lock = SourceLock.model_validate(
                _read_object(source_locks_path).get("sglang")
            )
            self.source_manager.create_worktree(source_lock, worktree)
            order = AgentWorkOrder(
                campaign_id=self.campaign_id,
                epoch=epoch,
                technique=technique,
                correctness=CorrectnessMode(entry.correctness),
                worktree=worktree.resolve(),
                delivery_path=(worktree / "DELIVERY.json").resolve(),
                review_path=(worktree / "AGENT-REVIEW.json").resolve(),
                baseline_path=baseline_path.resolve(),
                profile_path=profile_path.resolve(),
                technique_scope=entry.scope.resolve(),
                knowledge_manifest_path=knowledge_manifest_path.resolve(),
                search_space_path=search_space_path.resolve(),
                source_lock_sha256=_sha256_file(source_locks_path),
                baseline_sha256=_sha256_file(baseline_path),
                profile_sha256=_sha256_file(profile_path),
                technique_contract_sha256=_sha256_file(entry.scope),
                knowledge_manifest_sha256=_sha256_file(knowledge_manifest_path),
                search_space_sha256=_sha256_file(search_space_path),
                scientific_rounds_used=used,
                scientific_rounds_remaining=entry.round_budget - used,
            )
            _atomic_json(work_order_path, order.model_dump(mode="json"))
            actual_epoch = self.store.increment_epoch(
                self.campaign_id,
                idempotency_key=f"{self.campaign_id}:claim:{epoch}:epoch",
            )
            if actual_epoch != epoch:
                raise WorkOrderError(
                    f"campaign epoch changed while claiming work: {actual_epoch}"
                )
            self.store.transition(
                self.campaign_id,
                CampaignStatus.SEARCHING,
                idempotency_key=f"{self.campaign_id}:claim:{epoch}:searching",
                payload={
                    "epoch": epoch,
                    "technique": technique,
                    "work_order": str(work_order_path.resolve()),
                },
            )
            self.store.record_event(
                self.campaign_id,
                "work_claimed",
                f"{self.campaign_id}:claim:{epoch}:event",
                {
                    "epoch": epoch,
                    "technique": technique,
                    "work_order": str(work_order_path.resolve()),
                    "worktree": str(worktree.resolve()),
                },
            )
            return order

    def submit(self, delivery: Path) -> dict[str, Any]:
        with self._campaign_lock():
            order = self._require_active()
            supplied = delivery.expanduser().resolve()
            expected = order.delivery_path.resolve()
            if supplied != expected:
                raise WorkOrderError(
                    f"delivery must be the active work-order path: {expected}"
                )
            unresolved = order.delivery_path
            if unresolved.is_symlink() or not unresolved.is_file():
                raise WorkOrderError(
                    "delivery must be a regular, non-symlink file in the worktree"
                )
            try:
                supplied.relative_to(order.worktree.resolve())
            except ValueError as error:
                raise WorkOrderError("delivery escapes the active worktree") from error
            if self.store.events(
                self.campaign_id, event_type="candidate_submitted"
            ) and any(
                int(item["payload"].get("epoch", -1)) == order.epoch
                for item in self.store.events(
                    self.campaign_id, event_type="candidate_submitted"
                )
            ):
                raise WorkOrderError(
                    f"epoch {order.epoch} already has a candidate submission"
                )
            digest = _sha256_file(supplied)
            event = self.store.record_event(
                self.campaign_id,
                "candidate_submitted",
                f"{self.campaign_id}:submit:{order.epoch}",
                {
                    "epoch": order.epoch,
                    "technique": order.technique,
                    "delivery": str(supplied),
                    "delivery_sha256": digest,
                },
            )
            return event["payload"]

    def skip(
        self,
        technique: str,
        *,
        classification: Literal["unsupported", "no_gain", "blocked"],
        reason: str,
    ) -> TechniqueDisposition:
        reason = reason.strip()
        if not reason:
            raise WorkOrderError("skip reason must not be empty")
        with self._campaign_lock():
            status = self.store.status(self.campaign_id)
            if status not in {
                CampaignStatus.AWAITING_AGENT,
                CampaignStatus.SEARCHING,
            }:
                raise WorkOrderError(
                    "skip requires AWAITING_AGENT or SEARCHING, current status is "
                    f"{status.value}"
                )
            if technique not in self._routes():
                raise WorkOrderError(
                    f"technique {technique!r} is not a routed suggestion"
                )
            if status is CampaignStatus.SEARCHING:
                active = self._require_active()
                if active.technique != technique:
                    raise WorkOrderError(
                        f"active technique is {active.technique!r}, not {technique!r}"
                    )
            disposition = TechniqueDisposition(
                technique=technique,
                classification=classification,
                reason=reason,
                closed=classification in _CLOSED_CLASSIFICATIONS,
            )
            dispositions = self._dispositions()
            if (
                status is CampaignStatus.AWAITING_AGENT
                and dispositions.get(technique) == disposition
            ):
                return disposition
            dispositions[technique] = disposition
            _atomic_json(
                self.campaign_dir / "TECHNIQUE-DISPOSITIONS.json",
                {
                    "schema_version": 1,
                    "techniques": {
                        name: value.model_dump(mode="json")
                        for name, value in sorted(dispositions.items())
                    },
                },
            )
            epoch = self.store.epoch(self.campaign_id)
            self.store.record_event(
                self.campaign_id,
                "technique_skipped",
                (f"{self.campaign_id}:skip:{epoch}:{technique}:" f"{classification}"),
                {
                    "epoch": epoch,
                    **disposition.model_dump(mode="json"),
                },
            )
            if status is CampaignStatus.SEARCHING:
                self.store.transition(
                    self.campaign_id,
                    CampaignStatus.AWAITING_AGENT,
                    idempotency_key=(
                        f"{self.campaign_id}:skip:{epoch}:{technique}:awaiting"
                    ),
                    payload={
                        "technique": technique,
                        "classification": classification,
                    },
                )
            removed_verified, remaining_verified = self._verified_selection_after_skip(
                technique,
                exclude=disposition.closed,
            )
            if disposition.closed and removed_verified and remaining_verified:
                selection_epoch = self.store.increment_epoch(
                    self.campaign_id,
                    idempotency_key=(
                        f"{self.campaign_id}:skip:{epoch}:{technique}:"
                        "selection-epoch"
                    ),
                )
                self.store.transition(
                    self.campaign_id,
                    CampaignStatus.INTEGRATING,
                    idempotency_key=(
                        f"{self.campaign_id}:skip:{epoch}:{technique}:"
                        "reintegrate-subset"
                    ),
                    payload={
                        "excluded_technique": technique,
                        "selection_epoch": selection_epoch,
                        "reason": "reviewed_subset_changed",
                    },
                )
            elif self._all_routes_closed_or_exhausted():
                self.store.transition(
                    self.campaign_id,
                    CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                    idempotency_key=f"{self.campaign_id}:search-space-exhausted",
                    payload={"reason": "all_techniques_reviewed_or_budget_exhausted"},
                )
            return disposition

    def work(self) -> dict[str, Any]:
        with self._campaign_lock():
            status = self.store.status(self.campaign_id)
            routes = self._routes()
            dispositions = self._dispositions()
            active = (
                self._load_work_order(self.store.epoch(self.campaign_id))
                if status is CampaignStatus.SEARCHING
                else None
            )
            suggestions = []
            for name in routes:
                technique = self.registry[name]
                used = self._rounds_used(name)
                disposition = dispositions.get(name)
                suggestions.append(
                    {
                        "technique": name,
                        "correctness": technique.correctness,
                        "round_budget": technique.round_budget,
                        "scientific_rounds_used": used,
                        "scientific_rounds_remaining": max(
                            0, technique.round_budget - used
                        ),
                        "disposition": (
                            disposition.model_dump(mode="json")
                            if disposition is not None
                            else None
                        ),
                    }
                )
            return {
                "schema_version": 1,
                "execution_mode": "interactive_single_agent",
                "campaign_id": self.campaign_id,
                "status": status.value,
                "epoch": self.store.epoch(self.campaign_id),
                "active_work_order": (
                    active.model_dump(mode="json") if active is not None else None
                ),
                "suggestions": suggestions,
                "failures": self.store.failures(self.campaign_id),
                "legal_actions": self._legal_actions(status, active, suggestions),
            }

    def active_work_order(self) -> AgentWorkOrder:
        with self._campaign_lock():
            return self._require_active()

    def _require_active(self) -> AgentWorkOrder:
        status = self.store.status(self.campaign_id)
        if status is not CampaignStatus.SEARCHING:
            raise WorkOrderError(
                "no active work order; campaign status is " f"{status.value}"
            )
        return self._load_work_order(self.store.epoch(self.campaign_id))

    def _load_work_order(self, epoch: int) -> AgentWorkOrder:
        path = self.campaign_dir / "search" / str(epoch) / "AGENT-WORK.json"
        try:
            order = AgentWorkOrder.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise WorkOrderError(
                f"invalid active work order {path}: {error}"
            ) from error
        self._verify_work_order_bindings(order)
        return order

    def _verify_work_order_bindings(self, order: AgentWorkOrder) -> None:
        if order.technique not in self.registry.names():
            raise WorkOrderError(
                f"active work-order has unknown technique {order.technique!r}"
            )
        bindings = {
            "source lock": (
                self.campaign_dir / "SOURCE-LOCKS.json",
                order.source_lock_sha256,
            ),
            "baseline": (order.baseline_path, order.baseline_sha256),
            "profile": (order.profile_path, order.profile_sha256),
            "technique contract": (
                order.technique_scope,
                order.technique_contract_sha256,
            ),
            "knowledge manifest": (
                order.knowledge_manifest_path,
                order.knowledge_manifest_sha256,
            ),
            "search space": (
                order.search_space_path,
                order.search_space_sha256,
            ),
        }
        expected_paths = {
            "baseline": self.campaign_dir / "BASELINE.json",
            "profile": self.campaign_dir / "profiles" / "0" / "PROFILE-DIGEST.json",
            "technique contract": self.registry[order.technique].scope,
            "knowledge manifest": self.campaign_dir / "KNOWLEDGE.json",
            "search space": self.campaign_dir / "SEARCH-SPACE.json",
        }
        for label, (artifact, expected_digest) in bindings.items():
            resolved = artifact.resolve()
            expected_path = expected_paths.get(label)
            if expected_path is not None and resolved != expected_path.resolve():
                raise WorkOrderError(
                    f"active work-order {label} path differs from campaign binding"
                )
            if artifact.is_symlink() or not artifact.is_file():
                raise WorkOrderError(
                    f"active work-order {label} is not a regular artifact"
                )
            if _sha256_file(artifact) != expected_digest:
                raise WorkOrderError(
                    f"active work-order {label} hash differs from its binding"
                )

    def _routes(self) -> list[str]:
        value = _read_object(self.campaign_dir / "ROUTES.json")
        raw = value.get("routes")
        if not isinstance(raw, list) or not raw:
            raise WorkOrderError("ROUTES.json must contain non-empty routes")
        routes = [str(item) for item in raw]
        if len(routes) != len(set(routes)):
            raise WorkOrderError("ROUTES.json contains duplicate routes")
        unknown = set(routes) - set(self.registry.names())
        if unknown:
            raise WorkOrderError(
                "ROUTES.json contains unknown techniques: " + ", ".join(sorted(unknown))
            )
        return routes

    def _dispositions(self) -> dict[str, TechniqueDisposition]:
        path = self.campaign_dir / "TECHNIQUE-DISPOSITIONS.json"
        if not path.is_file():
            return {}
        raw = _read_object(path).get("techniques")
        if not isinstance(raw, dict):
            raise WorkOrderError("TECHNIQUE-DISPOSITIONS.json is malformed")
        return {
            str(name): TechniqueDisposition.model_validate(value)
            for name, value in raw.items()
        }

    def _rounds_used(self, technique: str) -> int:
        return sum(
            1
            for event in self.store.events(
                self.campaign_id, event_type="candidate_submitted"
            )
            if event["payload"].get("technique") == technique
        )

    def _all_routes_closed_or_exhausted(self) -> bool:
        dispositions = self._dispositions()
        for name in self._routes():
            disposition = dispositions.get(name)
            if disposition is not None and disposition.closed:
                continue
            if self._rounds_used(name) >= self.registry[name].round_budget:
                continue
            return False
        return True

    def _verified_selection_after_skip(
        self,
        technique: str,
        *,
        exclude: bool,
    ) -> tuple[bool, bool]:
        if not exclude:
            return False, False
        path = self.campaign_dir / "VERIFIED-CANDIDATES.json"
        if not path.is_file():
            return False, False
        candidates = _read_object(path).get("candidates")
        if not isinstance(candidates, dict):
            raise WorkOrderError("VERIFIED-CANDIDATES.json is malformed")
        dispositions = self._dispositions()

        def latency_positive(candidate: object) -> bool:
            if not isinstance(candidate, dict):
                return False
            speedup = candidate.get("verified_speedup")
            return (
                candidate.get("verified") is True
                and isinstance(speedup, (int, float))
                and not isinstance(speedup, bool)
                and float(speedup) > 1.0
            )

        def eligible(name: str, candidate: object) -> bool:
            disposition = dispositions.get(name)
            return latency_positive(candidate) and not (
                disposition is not None and disposition.closed
            )

        removed = latency_positive(candidates.get(technique))
        remaining = any(
            eligible(str(name), candidate)
            for name, candidate in candidates.items()
            if str(name) != technique
        )
        return removed, remaining

    @staticmethod
    def _legal_actions(
        status: CampaignStatus,
        active: AgentWorkOrder | None,
        suggestions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if status is CampaignStatus.AWAITING_AGENT:
            return [
                {
                    "action": "claim",
                    "technique": item["technique"],
                }
                for item in suggestions
                if not (
                    item["disposition"] is not None
                    and item["disposition"]["closed"] is True
                )
                and item["scientific_rounds_remaining"] > 0
            ] + [
                {"action": "skip", "technique": item["technique"]}
                for item in suggestions
            ]
        if status is CampaignStatus.SEARCHING and active is not None:
            return [
                {
                    "action": "submit",
                    "delivery": str(active.delivery_path),
                },
                {
                    "action": "skip",
                    "technique": active.technique,
                },
            ]
        return []
