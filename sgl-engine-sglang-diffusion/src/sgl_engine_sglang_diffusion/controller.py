from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import (
    BaselineRecord,
    CampaignGoal,
    CampaignStatus,
    UnreachableCertificate,
)
from .state import StateStore


class ControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class StepResult:
    """One evidence-backed phase decision returned to the state controller."""

    next_status: CampaignStatus | None
    payload: Mapping[str, Any] = field(default_factory=dict)
    verified_speedup: float | None = None
    clean_room_verified: bool = False
    new_hypothesis: bool = False
    failure_signature: str | None = None
    unreachable_certificate: Path | None = None


class CampaignHooks(Protocol):
    def freeze_sources_and_baseline(self) -> StepResult: ...

    def profile_and_route(self) -> StepResult: ...

    def start_search_epoch(self, epoch: int) -> StepResult: ...

    def poll_and_verify_executors(self, epoch: int) -> StepResult: ...

    def integrate_and_gate(self, epoch: int) -> StepResult: ...

    def package_or_continue(self, epoch: int) -> StepResult: ...


class CampaignController:
    """Durable state machine; scientific work is supplied by campaign hooks."""

    def __init__(
        self,
        *,
        store: StateStore,
        campaign_id: str,
        goal: CampaignGoal,
        hooks: CampaignHooks,
        campaign_dir: Path,
        allowed_methods: Sequence[str] = (),
    ) -> None:
        self.store = store
        self.campaign_id = campaign_id
        self.goal = goal
        self.hooks = hooks
        self.campaign_dir = campaign_dir.resolve()
        self.allowed_methods = tuple(allowed_methods)

    def run_once(self) -> CampaignStatus:
        status = self.store.status(self.campaign_id)
        epoch = self.store.epoch(self.campaign_id)
        self._heartbeat(status, epoch)
        if status is CampaignStatus.NEW:
            result = self.hooks.freeze_sources_and_baseline()
        elif status is CampaignStatus.BASELINE_LOCKED:
            result = self.hooks.profile_and_route()
        elif status is CampaignStatus.PROFILED:
            epoch = self.store.increment_epoch(
                self.campaign_id,
                idempotency_key=f"{self.campaign_id}:epoch:{epoch + 1}",
            )
            result = self.hooks.start_search_epoch(epoch)
        elif status is CampaignStatus.SEARCHING:
            result = self.hooks.poll_and_verify_executors(epoch)
        elif status is CampaignStatus.INTEGRATING:
            result = self.hooks.integrate_and_gate(epoch)
        elif status is CampaignStatus.FINAL_VERIFYING:
            result = self.hooks.package_or_continue(epoch)
        else:
            return status

        if result.next_status is None:
            self._heartbeat(status, epoch)
            return status
        normalized = self._normalize_result(status, epoch, result)
        if normalized.next_status is None:  # defensive for custom hook objects
            self._heartbeat(status, epoch)
            return status
        key = (
            f"{self.campaign_id}:{epoch}:{status.value}:"
            f"{normalized.next_status.value}"
        )
        payload = dict(normalized.payload)
        if normalized.verified_speedup is not None:
            payload["verified_speedup"] = normalized.verified_speedup
        payload["clean_room_verified"] = normalized.clean_room_verified
        next_status = self.store.transition(
            self.campaign_id,
            normalized.next_status,
            idempotency_key=key,
            payload=payload,
        )
        self._heartbeat(next_status, epoch)
        return next_status

    def run_until_wait(self, *, max_steps: int | None = None) -> CampaignStatus:
        """Advance until a terminal/recoverable state or hooks ask us to wait."""
        steps = 0
        while max_steps is None or steps < max_steps:
            previous = self.store.status(self.campaign_id)
            if previous not in {
                CampaignStatus.NEW,
                CampaignStatus.BASELINE_LOCKED,
                CampaignStatus.PROFILED,
                CampaignStatus.SEARCHING,
                CampaignStatus.INTEGRATING,
                CampaignStatus.FINAL_VERIFYING,
            }:
                return previous
            current = self.run_once()
            steps += 1
            if current in {
                CampaignStatus.WAITING_RESOURCE,
                CampaignStatus.INFRA_BLOCKED,
                CampaignStatus.PAUSED_BUDGET,
                CampaignStatus.TARGET_REACHED,
                CampaignStatus.UNREACHABLE_CERTIFIED,
                CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                CampaignStatus.CANCELLED,
            }:
                return current
        return self.store.status(self.campaign_id)

    def admit_hypothesis(
        self,
        *,
        technique: str,
        failure_signature: str | None,
        payload: Mapping[str, Any],
    ) -> bool:
        """Reject globally repeated failures before spending another GPU run."""
        if failure_signature is None:
            return True
        return self.store.record_failure(
            self.campaign_id,
            technique,
            failure_signature,
            payload,
        )

    def _normalize_result(
        self,
        current: CampaignStatus,
        epoch: int,
        result: StepResult,
    ) -> StepResult:
        target = result.next_status
        if target is CampaignStatus.TARGET_REACHED:
            if (
                result.verified_speedup is None
                or result.verified_speedup < self.goal.goal.target_speedup
                or not result.clean_room_verified
            ):
                if result.new_hypothesis:
                    return StepResult(
                        CampaignStatus.PROFILED,
                        payload={
                            **dict(result.payload),
                            "reason": "target_not_reached",
                        },
                    )
                return StepResult(
                    CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                    payload={
                        **dict(result.payload),
                        "reason": "unverified_or_below_target",
                    },
                )
        if target is CampaignStatus.UNREACHABLE_CERTIFIED:
            if result.unreachable_certificate is None or not self._valid_unreachable(
                result.unreachable_certificate
            ):
                return StepResult(
                    CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                    payload={
                        **dict(result.payload),
                        "reason": "missing_or_invalid_lower_bound_certificate",
                    },
                )
        if target is current:
            raise ControllerError(
                f"{current.value} cannot self-transition; hooks must return "
                "next_status=None to wait"
            )
        if result.failure_signature:
            admitted = self.admit_hypothesis(
                technique=str(result.payload.get("technique", "unknown")),
                failure_signature=result.failure_signature,
                payload=result.payload,
            )
            if not admitted and result.new_hypothesis:
                return StepResult(
                    CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                    payload={
                        **dict(result.payload),
                        "reason": "repeated_failure_signature",
                    },
                )
        return result

    def _valid_unreachable(self, path: Path) -> bool:
        try:
            certificate = UnreachableCertificate.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            baseline = BaselineRecord.model_validate_json(
                (self.campaign_dir / "BASELINE.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return False
        prompt_hash = hashlib.sha256(
            self.goal.workload.prompts.read_bytes()
        ).hexdigest()
        target_latency = baseline.mean_e2e_s / self.goal.goal.target_speedup
        return (
            certificate.frozen_workload_sha256 == prompt_hash
            and certificate.hardware == self.goal.hardware.model_dump(mode="json")
            and set(certificate.allowed_methods) == set(self.allowed_methods)
            and abs(certificate.target_latency_s - target_latency)
            <= max(1e-9, target_latency * 1e-9)
            and certificate.lower_bound_s > certificate.target_latency_s
        )

    def _heartbeat(self, status: CampaignStatus, epoch: int) -> None:
        target = self.campaign_dir / "controller-heartbeat.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": self.campaign_id,
                    "status": status.value,
                    "epoch": epoch,
                    "pid": os.getpid(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
