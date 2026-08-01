from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.baseline import BaselineError, BaselineRunner
from sgl_engine_sglang_diffusion.driver import (
    Activation,
    DriverError,
    SGLangDiffusionDriver,
)
from sgl_engine_sglang_diffusion.models import CampaignGoal
from sgl_engine_sglang_diffusion.process import CommandResult


COMMIT = "a" * 40


def make_goal(tmp_path: Path) -> CampaignGoal:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    return CampaignGoal.model_validate(
        {
            "schema_version": 1,
            "model": {"id": "test/model"},
            "hardware": {"environment": "test", "gpu_count": 1},
            "workload": {
                "prompts": prompts,
                "prompt_count": 5,
                "seed": 42,
                "height": 64,
                "width": 64,
                "frames": 1,
                "fps": 24,
                "steps": 4,
                "guidance": 1.0,
                "dtype": "bfloat16",
                "timing_scope": "load_excluded_end_to_end",
            },
            "goal": {"target_speedup": 2.0, "allow_quality_gated": True},
            "source": {"sglang_repo": "local", "sglang_ref": "main"},
            "agent": {"command": ["codex"]},
        }
    )


def make_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "sglang"
    benchmark = (
        checkout / "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py"
    )
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("# fake benchmark\n")
    return checkout


class FakeRunner:
    def __init__(self, *, fallback: bool = False) -> None:
        self.calls = 0
        self.fallback = fallback

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> CommandResult:
        self.calls += 1
        output = Path(argv[argv.index("--output-file") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "results": {
                        "successful_requests": 5,
                        "failed_requests": 0,
                        "total_duration_seconds": 400.0,
                        "latency_per_request_seconds": 80.0,
                        "peak_memory_mb": 1024.0,
                    }
                }
            )
            + "\n"
        )
        media = Path(argv[argv.index("--output-path") + 1])
        media.mkdir(parents=True, exist_ok=True)
        for index in range(5):
            (media / f"{index}.png").write_bytes(b"image")
        stdout = "loaded diffusers pipeline" if self.fallback else "native engine"
        return CommandResult(tuple(argv), 0, stdout, "")


def test_driver_builds_frozen_argv_and_profile_env(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    driver = SGLangDiffusionDriver(make_checkout(tmp_path))
    command = driver.build_command(
        goal,
        tmp_path / "run",
        activation=Activation(
            env={"SGLANG_AGENT_PROFILE": "candidate"},
            server_args=("--enable-torch-compile",),
        ),
        profile=True,
    )
    joined = "\0".join(command.argv)
    pairs = {
        "--model-path": "test/model",
        "--dataset": "vbench",
        "--dataset-path": str(goal.workload.prompts.resolve()),
        "--num-prompts": "5",
        "--seed": "42",
        "--width": "64",
        "--height": "64",
        "--num-frames": "1",
        "--fps": "24",
        "--num-inference-steps": "4",
        "--guidance-scale": "1.0",
        "--num-profiled-timesteps": "5",
    }
    for flag, value in pairs.items():
        expected = f"{flag}\0{value}"
        assert expected in joined
    assert "--profile" in command.argv
    assert "--enable-torch-compile" in command.argv
    assert command.env["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"] == str(
        (tmp_path / "run").resolve()
    )


def test_driver_rejects_fallback_and_redacts_receipt(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    runner = FakeRunner(fallback=True)
    driver = SGLangDiffusionDriver(make_checkout(tmp_path), runner=runner)
    run_dir = tmp_path / "run"
    with pytest.raises(DriverError, match="non-native"):
        driver.run(
            goal,
            run_dir,
            activation=Activation(env={"API_TOKEN": "do-not-record"}),
        )
    assert json.loads((run_dir / "COMMAND.json").read_text())["declared_env"] == {
        "API_TOKEN": "<redacted>"
    }


def test_baseline_is_frozen_once(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    runner = FakeRunner()
    driver = SGLangDiffusionDriver(make_checkout(tmp_path), runner=runner)
    baseline = BaselineRunner(driver)
    campaign = tmp_path / "campaign"

    record = baseline.freeze(goal, campaign, sglang_commit=COMMIT)
    assert record.mean_e2e_s == 80.0
    assert record.workload_total_s == 400.0
    assert record.request_count == 5
    assert len(list(record.baseline_frames.glob("prompt-*"))) == 5
    assert runner.calls == 1

    with pytest.raises(BaselineError, match="cannot be refreshed"):
        baseline.freeze(goal, campaign, sglang_commit=COMMIT)
    assert runner.calls == 1


def test_fallback_never_writes_baseline(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    driver = SGLangDiffusionDriver(
        make_checkout(tmp_path), runner=FakeRunner(fallback=True)
    )
    campaign = tmp_path / "campaign"
    with pytest.raises(BaselineError, match="non-native"):
        BaselineRunner(driver).freeze(goal, campaign, sglang_commit=COMMIT)
    assert not (campaign / "BASELINE.json").exists()


def test_failed_baseline_attempt_can_retry_without_overwriting(
    tmp_path: Path,
) -> None:
    goal = make_goal(tmp_path)
    runner = FakeRunner(fallback=True)
    driver = SGLangDiffusionDriver(make_checkout(tmp_path), runner=runner)
    campaign = tmp_path / "campaign"
    with pytest.raises(BaselineError):
        BaselineRunner(driver).freeze(goal, campaign, sglang_commit=COMMIT)

    runner.fallback = False
    record = BaselineRunner(driver).freeze(goal, campaign, sglang_commit=COMMIT)
    assert record.run_dir.name == "attempt-002"
    assert (campaign / "baseline/attempt-001/COMMAND.json").is_file()
    assert (campaign / "baseline/attempt-002/COMMAND.json").is_file()


def test_normalizer_rejects_failed_requests(tmp_path: Path) -> None:
    result = tmp_path / "benchmark.jsonl"
    result.write_text(
        json.dumps(
            {
                "results": {
                    "successful_requests": 4,
                    "failed_requests": 1,
                    "total_duration_seconds": 1,
                    "peak_memory_mb": 1,
                }
            }
        )
        + "\n"
    )
    with pytest.raises(DriverError, match="successful requests"):
        SGLangDiffusionDriver.normalize_output(
            result, timing_scope="end_to_end", expected_requests=5
        )


def test_normalizer_uses_per_request_mean_as_authoritative_latency(
    tmp_path: Path,
) -> None:
    result = tmp_path / "benchmark.jsonl"
    result.write_text(
        json.dumps(
            {
                "results": {
                    "successful_requests": 5,
                    "failed_requests": 0,
                    "total_duration_seconds": 400.0,
                    "latency_per_request_seconds": 80.0,
                    "peak_memory_mb": 1024.0,
                }
            }
        )
        + "\n"
    )

    normalized = SGLangDiffusionDriver.normalize_output(
        result, timing_scope="end_to_end", expected_requests=5
    )

    assert normalized["schema_version"] == 2
    assert normalized["mean_e2e_s"] == 80.0
    assert normalized["workload_total_s"] == 400.0
    assert normalized["request_count"] == 5
    assert "total_s" not in normalized


def test_normalizer_rejects_inconsistent_reported_request_mean(
    tmp_path: Path,
) -> None:
    result = tmp_path / "benchmark.jsonl"
    result.write_text(
        json.dumps(
            {
                "results": {
                    "successful_requests": 5,
                    "failed_requests": 0,
                    "total_duration_seconds": 400.0,
                    "latency_per_request_seconds": 79.0,
                    "peak_memory_mb": 1024.0,
                }
            }
        )
        + "\n"
    )

    with pytest.raises(DriverError, match="per-request latency disagrees"):
        SGLangDiffusionDriver.normalize_output(
            result, timing_scope="end_to_end", expected_requests=5
        )


def test_normalizer_requires_successful_request_count(tmp_path: Path) -> None:
    result = tmp_path / "benchmark.jsonl"
    result.write_text(
        json.dumps(
            {
                "results": {
                    "failed_requests": 0,
                    "total_duration_seconds": 400.0,
                    "peak_memory_mb": 1024.0,
                }
            }
        )
        + "\n"
    )

    with pytest.raises(DriverError, match="successful request count"):
        SGLangDiffusionDriver.normalize_output(
            result, timing_scope="end_to_end", expected_requests=5
        )
