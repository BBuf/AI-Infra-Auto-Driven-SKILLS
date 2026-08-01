from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from sgl_engine_sglang_diffusion.models import (
    BaselineRecord,
    CampaignGoal,
    FinalQualityEvidence,
)
from sgl_engine_sglang_diffusion.runtime import FileCampaignHooks

from test_driver import make_goal


def _quality_payload(tmp_path: Path, goal: CampaignGoal) -> dict[str, object]:
    receipts = []
    for name in ("lpips", "vbench", "audio", "media-avsync"):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"tool": "{name}"}}\n')
        receipts.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    dimensions = {
        "subject_consistency": 0.7,
        "background_consistency": 0.7,
        "motion_smoothness": 0.7,
        "temporal_flickering": 0.7,
        "aesthetic_quality": 0.7,
        "imaging_quality": 0.7,
    }
    prompts = []
    for index in range(5):
        prompts.append(
            {
                "prompt_index": index,
                "lpips": 0.05,
                "vbench_baseline": dict(dimensions),
                "vbench_candidate": dict(dimensions),
                "audio": {
                    "present": False,
                    "duration_s": 0.0,
                    "sample_rate": 0,
                    "channels": 0,
                    "silence_ratio": 0.0,
                    "clipping_ratio": 0.0,
                },
                "av_sync_drift_ms": None,
                "media": {
                    "container": "png",
                    "video_codec": "png",
                    "audio_codec": None,
                    "width": goal.workload.width,
                    "height": goal.workload.height,
                    "fps": goal.workload.fps,
                    "frame_count": goal.workload.frames,
                    "video_duration_s": goal.workload.frames / goal.workload.fps,
                },
                "visual": "pass",
            }
        )
    return {
        "schema_version": 1,
        "producer": "independent-master",
        "external_api": False,
        "integrated_commit": "a" * 40,
        "audio_required": False,
        "thresholds": {
            "lpips_max": 0.1,
            "vbench_max_mean_regression": 0.0,
            "silence_ratio_max": 0.98,
            "clipping_ratio_max": 0.01,
            "av_sync_drift_ms_max": 80.0,
        },
        "prompts": prompts,
        "command_receipts": receipts,
    }


def _hooks_and_baseline(tmp_path: Path) -> tuple[FileCampaignHooks, BaselineRecord]:
    goal = make_goal(tmp_path)
    hooks = object.__new__(FileCampaignHooks)
    hooks.goal = goal
    hooks.campaign_dir = tmp_path
    run_dir = tmp_path / "baseline-run"
    run_dir.mkdir()
    for index in range(5):
        (run_dir / f"prompt-{index}.png").write_bytes(b"image")
    baseline = BaselineRecord(
        model_id=goal.model.id,
        total_s=10.0,
        peak_memory_mib=100.0,
        timing_scope=goal.workload.timing_scope,
        run_dir=run_dir,
        baseline_frames=run_dir,
        sglang_commit="b" * 40,
    )
    return hooks, baseline


def test_complete_five_prompt_quality_evidence_passes(tmp_path: Path) -> None:
    hooks, baseline = _hooks_and_baseline(tmp_path)
    evidence = FinalQualityEvidence.model_validate(_quality_payload(tmp_path, hooks.goal))
    assert hooks._validate_final_quality_evidence(
        evidence,
        integrated_commit="a" * 40,
        baseline=baseline,
        evidence_root=tmp_path,
    ) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value["prompts"][0].update(lpips=0.2), "LPIPS"),
        (lambda value: value["prompts"][0].update(visual="fail"), "visual"),
        (
            lambda value: value["prompts"][0]["media"].update(width=1),
            "media contract",
        ),
        (
            lambda value: value["prompts"][0]["vbench_candidate"].update(
                subject_consistency=0.0
            ),
            "VBench",
        ),
    ],
)
def test_final_quality_sections_fail_closed(
    tmp_path: Path, mutation: object, expected: str
) -> None:
    hooks, baseline = _hooks_and_baseline(tmp_path)
    payload = _quality_payload(tmp_path, hooks.goal)
    mutation(payload)  # type: ignore[operator]
    evidence = FinalQualityEvidence.model_validate(payload)
    issues = hooks._validate_final_quality_evidence(
        evidence,
        integrated_commit="a" * 40,
        baseline=baseline,
        evidence_root=tmp_path,
    )
    assert expected in " ".join(issues)


def test_missing_prompt_or_tool_receipts_are_rejected(tmp_path: Path) -> None:
    hooks, _ = _hooks_and_baseline(tmp_path)
    payload = _quality_payload(tmp_path, hooks.goal)
    payload["prompts"] = payload["prompts"][:4]  # type: ignore[index]
    with pytest.raises(ValidationError, match="at least 5 items"):
        FinalQualityEvidence.model_validate(payload)

    payload = _quality_payload(tmp_path, hooks.goal)
    payload["command_receipts"] = payload["command_receipts"][:3]  # type: ignore[index]
    with pytest.raises(ValidationError, match="at least 4 items"):
        FinalQualityEvidence.model_validate(payload)


def test_audio_and_av_sync_are_required_when_baseline_has_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks, baseline = _hooks_and_baseline(tmp_path)
    monkeypatch.setattr(
        FileCampaignHooks,
        "_baseline_audio_required",
        staticmethod(lambda _: True),
    )
    evidence = FinalQualityEvidence.model_validate(_quality_payload(tmp_path, hooks.goal))
    issues = hooks._validate_final_quality_evidence(
        evidence,
        integrated_commit="a" * 40,
        baseline=baseline,
        evidence_root=tmp_path,
    )
    assert "audio requirement" in " ".join(issues)
    assert "audio quality" in " ".join(issues)
    assert "AV sync" in " ".join(issues)
