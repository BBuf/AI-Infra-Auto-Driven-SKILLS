from __future__ import annotations

from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.request import (
    AgentSpec,
    BaselineCommandRequest,
    LaunchRequest,
    RequestError,
    normalize_launch_request,
)


def make_request(tmp_path: Path, command: str) -> LaunchRequest:
    checkout = tmp_path / "sglang"
    benchmark = (
        checkout / "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py"
    )
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text("# benchmark\n")
    prompts = checkout / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    return LaunchRequest(
        machine="test-b200",
        model="test/model",
        sglang_checkout=checkout,
        target_speedup=2.0,
        baseline=BaselineCommandRequest(command=command, cwd=checkout),
        run_root=tmp_path / "runs",
    )


def native_command() -> str:
    return (
        "CUDA_VISIBLE_DEVICES=0 python "
        "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py "
        "--model-path test/model --dataset vbench --dataset-path prompts.txt "
        "--num-prompts 5 --seed 7 --width 64 --height 32 --num-frames 9 "
        "--fps 12 --num-inference-steps 4 --guidance-scale 1.5 "
        "--dtype bfloat16 --output-file old.json"
    )


def test_normalize_command_extracts_goal_and_relocatable_template(
    tmp_path: Path,
) -> None:
    goal, command = normalize_launch_request(make_request(tmp_path, native_command()))
    assert goal.schema_version == 2
    assert goal.execution_mode == "interactive_single_agent"
    assert not hasattr(goal, "agent")
    assert goal.model.id == "test/model"
    assert goal.hardware.environment == "test-b200"
    assert goal.workload.seed == 7
    assert goal.workload.width == 64
    assert goal.workload.frames == 9
    assert command.env["CUDA_VISIBLE_DEVICES"] == "0"

    other_checkout = tmp_path / "candidate"
    benchmark = (
        other_checkout
        / "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py"
    )
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("# candidate\n")
    argv, environment = command.render(
        checkout=other_checkout,
        prompts=goal.workload.prompts,
        output_file=tmp_path / "out.jsonl",
        media_dir=tmp_path / "media",
        activation_env={"SGLANG_AGENT_PROFILE": "candidate"},
        activation_args=("--enable-fast-path",),
    )
    assert str(benchmark) in argv
    assert argv[argv.index("--dataset-path") + 1] == str(
        goal.workload.prompts.resolve()
    )
    assert argv[argv.index("--output-file") + 1] == str(
        (tmp_path / "out.jsonl").resolve()
    )
    assert "--enable-fast-path" in argv
    assert environment["SGLANG_AGENT_PROFILE"] == "candidate"
    assert environment["PYTHONPATH"].startswith(str(other_checkout / "python"))


def test_legacy_agent_request_is_accepted_but_not_frozen(tmp_path: Path) -> None:
    request = make_request(tmp_path, native_command())
    request.agent = AgentSpec(command=["codex", "exec"], model="legacy-model")

    goal, _ = normalize_launch_request(request)

    assert goal.execution_mode == "interactive_single_agent"
    assert "agent" not in goal.model_dump(mode="json")


@pytest.mark.parametrize(
    "command",
    [
        f"{native_command()} | tee result.log",
        f"{native_command()} && echo done",
        f"{native_command()} > result.log",
        f"{native_command()} 2>&1",
        f"{native_command()} --model-path $(whoami)",
        f"{native_command()} --dataset-path $PROMPTS",
        f"{native_command()} `whoami`",
    ],
)
def test_normalize_rejects_shell_features(tmp_path: Path, command: str) -> None:
    with pytest.raises(RequestError, match="shell|substitution"):
        normalize_launch_request(make_request(tmp_path, command))


def test_normalize_rejects_model_or_prompt_contract_drift(tmp_path: Path) -> None:
    request = make_request(
        tmp_path,
        native_command().replace("--model-path test/model", "--model-path other"),
    )
    with pytest.raises(RequestError, match="model differs"):
        normalize_launch_request(request)

    request = make_request(
        tmp_path,
        native_command().replace("--num-prompts 5", "--num-prompts 4"),
    )
    with pytest.raises(RequestError, match="five"):
        normalize_launch_request(request)


def test_module_entrypoint_is_supported(tmp_path: Path) -> None:
    command = native_command().replace(
        "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py",
        "-m sglang.multimodal_gen.benchmarks.bench_offline_throughput",
    )
    _, template = normalize_launch_request(make_request(tmp_path, command))
    assert template.mode == "module"
    assert "-m" in template.argv_template


def test_secret_environment_values_are_not_persisted(tmp_path: Path) -> None:
    request = make_request(
        tmp_path,
        f"HF_TOKEN=do-not-store {native_command()}",
    )
    with pytest.raises(RequestError, match="inherit them"):
        normalize_launch_request(request)

    request = make_request(
        tmp_path,
        f"{native_command()} --api-key do-not-store",
    )
    with pytest.raises(RequestError, match="baseline argv"):
        normalize_launch_request(request)


def test_missing_optional_flags_use_locked_defaults_without_injection(
    tmp_path: Path,
) -> None:
    command = (
        "python "
        "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py "
        "--model-path test/model --dataset vbench --dataset-path prompts.txt "
        "--num-prompts 5"
    )
    goal, template = normalize_launch_request(make_request(tmp_path, command))
    assert goal.workload.dtype == "auto"
    assert goal.workload.steps == 20
    assert "--dtype" not in template.argv_template
    assert "--num-inference-steps" not in template.argv_template
    with pytest.raises(RequestError, match="changed frozen flag"):
        template.render(
            checkout=tmp_path / "sglang",
            prompts=goal.workload.prompts,
            output_file=tmp_path / "output.jsonl",
            media_dir=tmp_path / "media",
            activation_args=("--num-inference-steps", "1"),
        )
    with pytest.raises(RequestError, match="frozen environment"):
        template.render(
            checkout=tmp_path / "sglang",
            prompts=goal.workload.prompts,
            output_file=tmp_path / "output.jsonl",
            media_dir=tmp_path / "media",
            activation_env={"PYTHONPATH": "/another/checkout"},
        )


def test_baseline_cwd_and_script_must_match_checkout(tmp_path: Path) -> None:
    request = make_request(tmp_path, native_command())
    request.baseline.cwd = tmp_path
    with pytest.raises(RequestError, match="checkout root"):
        normalize_launch_request(request)

    request = make_request(
        tmp_path,
        native_command().replace(
            "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py",
            "/tmp/bench_offline_throughput.py",
        ),
    )
    with pytest.raises(RequestError, match="not from"):
        normalize_launch_request(request)
