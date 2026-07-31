from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import CandidateManifest, StrictModel
from .process import run


class ReviewError(RuntimeError):
    """A same-agent review is missing, stale, or bound to different evidence."""


class VisualReviewBinding(StrictModel):
    required: bool
    accepted: bool
    prompt_count: int = Field(ge=0)
    artifact_sha256: list[str]


class AgentReview(StrictModel):
    schema_version: Literal[1] = 1
    producer: Literal["interactive-root-agent"]
    campaign_id: str = Field(min_length=1)
    epoch: int = Field(ge=1)
    technique: str = Field(min_length=1)
    baseline_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_argument_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_equivalent: bool
    accepted: bool
    visual_review: VisualReviewBinding
    findings: list[str]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_diff_sha256(
    worktree: Path,
    *,
    baseline_commit: str,
    candidate_commit: str,
) -> str:
    result = run(
        [
            "git",
            "diff",
            "--binary",
            "--full-index",
            baseline_commit,
            candidate_commit,
            "--",
        ],
        cwd=worktree,
    )
    return sha256_text(result.stdout)


class SameAgentReviewValidator:
    """Validate the root agent's verdict against the exact submitted commits."""

    def __init__(
        self,
        *,
        campaign_id: str,
        epoch: int,
        review_path: Path,
    ) -> None:
        self.campaign_id = campaign_id
        self.epoch = epoch
        self.review_path = review_path.resolve()

    def validate(
        self,
        *,
        technique: str,
        candidate_worktree: Path,
        manifest: CandidateManifest,
        method_argument: str,
        visual_verdict_path: Path,
        quality_gated: bool,
    ) -> tuple[str, ...]:
        worktree = candidate_worktree.resolve()
        expected_review = (worktree / "AGENT-REVIEW.json").resolve()
        if self.review_path != expected_review:
            raise ReviewError(
                "same-agent review must be the active worktree AGENT-REVIEW.json"
            )
        unresolved = worktree / "AGENT-REVIEW.json"
        if unresolved.is_symlink() or not unresolved.is_file():
            raise ReviewError("same-agent review must be a regular, non-symlink file")
        review = AgentReview.model_validate_json(unresolved.read_text(encoding="utf-8"))
        mismatches: list[str] = []
        expected = {
            "campaign_id": self.campaign_id,
            "epoch": self.epoch,
            "technique": technique,
            "baseline_commit": manifest.base_commit,
            "candidate_commit": manifest.candidate_commit,
            "diff_sha256": git_diff_sha256(
                worktree,
                baseline_commit=manifest.base_commit,
                candidate_commit=manifest.candidate_commit,
            ),
            "method_argument_sha256": sha256_text(method_argument),
        }
        for name, value in expected.items():
            if getattr(review, name) != value:
                mismatches.append(f"{name} does not match submitted evidence")
        if not review.method_equivalent:
            mismatches.append("method_equivalent is false")
        if not review.accepted:
            mismatches.append("same-agent review did not accept the candidate")
        if review.findings:
            mismatches.append("same-agent review contains unresolved findings")

        verdict_digest = sha256_file(visual_verdict_path)
        visual = review.visual_review
        if quality_gated:
            if not visual.required:
                mismatches.append("quality-gated work requires visual review")
            if not visual.accepted:
                mismatches.append("same-agent visual review did not pass")
            if visual.prompt_count != 5:
                mismatches.append("same-agent visual review must cover five prompts")
            if verdict_digest not in visual.artifact_sha256:
                mismatches.append("visual verdict digest is not bound to the review")
        else:
            if visual.required:
                mismatches.append("lossless work must not claim a quality gate")
            if not visual.accepted:
                mismatches.append("lossless authenticity review did not pass")
            if verdict_digest not in visual.artifact_sha256:
                mismatches.append("authenticity verdict digest is not bound to review")
        return tuple(mismatches)
