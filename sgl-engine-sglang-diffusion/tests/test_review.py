from __future__ import annotations

import json
from pathlib import Path

from sgl_engine_sglang_diffusion.models import CandidateManifest
from sgl_engine_sglang_diffusion.process import run
from sgl_engine_sglang_diffusion.review import (
    SameAgentReviewValidator,
    git_diff_sha256,
    sha256_file,
    sha256_text,
)


def _review_fixture(tmp_path: Path) -> tuple[Path, CandidateManifest, Path, str]:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    run(["git", "init"], cwd=worktree)
    run(["git", "config", "user.email", "test@example.invalid"], cwd=worktree)
    run(["git", "config", "user.name", "Test"], cwd=worktree)
    source = worktree / "kernel.py"
    source.write_text("value = 1\n", encoding="utf-8")
    run(["git", "add", "kernel.py"], cwd=worktree)
    run(["git", "commit", "-m", "baseline"], cwd=worktree)
    baseline = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    source.write_text("value = 2\n", encoding="utf-8")
    run(["git", "add", "kernel.py"], cwd=worktree)
    run(["git", "commit", "-m", "candidate"], cwd=worktree)
    candidate = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    manifest = CandidateManifest(
        candidate_id="candidate",
        technique="kernel",
        kind="patch",
        base_commit=baseline,
        candidate_commit=candidate,
        activation={"enabled": True},
        eval_profile={"timing_scope": "frozen"},
        knowledge_origin=[],
    )
    verdict = worktree / "authenticity.json"
    verdict.write_text('{"overall":"authenticity_only"}\n', encoding="utf-8")
    method_argument = "The indexed contraction is unchanged."
    review = {
        "schema_version": 1,
        "producer": "interactive-root-agent",
        "campaign_id": "campaign",
        "epoch": 1,
        "technique": "kernel",
        "baseline_commit": baseline,
        "candidate_commit": candidate,
        "diff_sha256": git_diff_sha256(
            worktree,
            baseline_commit=baseline,
            candidate_commit=candidate,
        ),
        "method_argument_sha256": sha256_text(method_argument),
        "method_equivalent": True,
        "accepted": True,
        "visual_review": {
            "required": False,
            "accepted": True,
            "prompt_count": 0,
            "artifact_sha256": [sha256_file(verdict)],
        },
        "findings": [],
    }
    (worktree / "AGENT-REVIEW.json").write_text(json.dumps(review), encoding="utf-8")
    return worktree, manifest, verdict, method_argument


def test_same_agent_review_is_bound_to_exact_commits_and_method(
    tmp_path: Path,
) -> None:
    worktree, manifest, verdict, method_argument = _review_fixture(tmp_path)
    validator = SameAgentReviewValidator(
        campaign_id="campaign",
        epoch=1,
        review_path=worktree / "AGENT-REVIEW.json",
    )

    assert (
        validator.validate(
            technique="kernel",
            candidate_worktree=worktree,
            manifest=manifest,
            method_argument=method_argument,
            visual_verdict_path=verdict,
            quality_gated=False,
        )
        == ()
    )


def test_stale_review_diff_and_method_are_rejected(tmp_path: Path) -> None:
    worktree, manifest, verdict, method_argument = _review_fixture(tmp_path)
    path = worktree / "AGENT-REVIEW.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    review["diff_sha256"] = "0" * 64
    path.write_text(json.dumps(review), encoding="utf-8")
    validator = SameAgentReviewValidator(
        campaign_id="campaign",
        epoch=1,
        review_path=path,
    )

    issues = validator.validate(
        technique="kernel",
        candidate_worktree=worktree,
        manifest=manifest,
        method_argument=method_argument + " drift",
        visual_verdict_path=verdict,
        quality_gated=False,
    )

    assert "diff_sha256 does not match submitted evidence" in issues
    assert "method_argument_sha256 does not match submitted evidence" in issues


def test_quality_review_requires_five_bound_prompt_artifacts(tmp_path: Path) -> None:
    worktree, manifest, verdict, _ = _review_fixture(tmp_path)
    path = worktree / "AGENT-REVIEW.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    review["technique"] = "cache"
    review["visual_review"] = {
        "required": True,
        "accepted": True,
        "prompt_count": 4,
        "artifact_sha256": [],
    }
    method_argument = json.dumps(
        {"activation": manifest.activation, "technique": "cache"},
        sort_keys=True,
        separators=(",", ":"),
    )
    review["method_argument_sha256"] = sha256_text(method_argument)
    path.write_text(json.dumps(review), encoding="utf-8")
    manifest = manifest.model_copy(update={"technique": "cache"})
    validator = SameAgentReviewValidator(
        campaign_id="campaign",
        epoch=1,
        review_path=path,
    )

    issues = validator.validate(
        technique="cache",
        candidate_worktree=worktree,
        manifest=manifest,
        method_argument=method_argument,
        visual_verdict_path=verdict,
        quality_gated=True,
    )

    assert "same-agent visual review must cover five prompts" in issues
    assert "visual verdict digest is not bound to the review" in issues
