from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Literal, Mapping, Sequence

import yaml
from pydantic import Field, model_validator

from .models import (
    AgentSpec,
    CampaignGoal,
    GoalTarget,
    HardwareSpec,
    ModelSpec,
    SourceSpec,
    StrictModel,
    WorkloadSpec,
)


class RequestError(ValueError):
    """A launch request cannot be converted into a safe frozen campaign."""


_BENCHMARK_MODULE = "sglang.multimodal_gen.benchmarks.bench_offline_throughput"
_BENCHMARK_RELATIVE = (
    "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py"
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_NAME = re.compile(r"(?:token|secret|password|key|credential)", re.IGNORECASE)
_UNSAFE_TOKEN = re.compile(r"(?:\$\(|\$\{|\$[A-Za-z_]|`|\n|\r)")
_PLACEHOLDERS = {
    "benchmark": "{{benchmark}}",
    "checkout": "{{checkout}}",
    "output_file": "{{output_file}}",
    "media_dir": "{{media_dir}}",
    "prompts": "{{prompts}}",
}
_VALUE_FLAGS = {
    "--model-path",
    "--dataset",
    "--dataset-path",
    "--num-prompts",
    "--seed",
    "--width",
    "--height",
    "--num-frames",
    "--fps",
    "--num-inference-steps",
    "--guidance-scale",
    "--dtype",
    "--output-path",
    "--output-file",
}
_FROZEN_FLAGS = {
    "--model-path",
    "--dataset",
    "--dataset-path",
    "--num-prompts",
    "--seed",
    "--width",
    "--height",
    "--num-frames",
    "--fps",
    "--num-inference-steps",
    "--guidance-scale",
    "--dtype",
}


class BaselineCommandRequest(StrictModel):
    command: str | None = None
    argv: list[str] | None = None
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path
    timing_scope: str = "load_excluded_end_to_end"

    @model_validator(mode="after")
    def require_one_command_form(self) -> BaselineCommandRequest:
        if (self.command is None) == (self.argv is None):
            raise ValueError("provide exactly one of baseline.command or baseline.argv")
        if self.command is not None and not self.command.strip():
            raise ValueError("baseline.command must not be empty")
        if self.argv is not None and (
            not self.argv or any(not item for item in self.argv)
        ):
            raise ValueError("baseline.argv must contain nonempty strings")
        return self


class LaunchRequest(StrictModel):
    schema_version: Literal[1] = 1
    machine: str = Field(min_length=1)
    model: str = Field(min_length=1)
    sglang_checkout: Path
    sglang_ref: str = "main"
    gpu_count: int = Field(default=1, ge=1)
    target_speedup: float = Field(gt=1.0)
    allow_quality_gated: bool = True
    baseline: BaselineCommandRequest
    agent: AgentSpec = Field(
        default_factory=lambda: AgentSpec(command=["codex", "exec"])
    )
    token_budget: int | None = Field(default=None, gt=0)
    run_root: Path = Path("runs/sglang-diffusion-auto-optimize")
    idempotency_key: str | None = None
    source: SourceSpec | None = None


class FrozenBenchmarkCommand(StrictModel):
    schema_version: Literal[1] = 1
    adapter: Literal["sglang_diffusion_offline"]
    mode: Literal["script", "module"]
    argv_template: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    original_cwd: Path
    original_command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_flags: dict[str, str]

    def render(
        self,
        *,
        checkout: Path,
        prompts: Path,
        output_file: Path,
        media_dir: Path,
        activation_env: Mapping[str, str] | None = None,
        activation_args: Sequence[str] = (),
        profile: bool = False,
        profile_dir: Path | None = None,
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        checkout = checkout.resolve()
        values = {
            _PLACEHOLDERS["benchmark"]: str(checkout / _BENCHMARK_RELATIVE),
            _PLACEHOLDERS["checkout"]: str(checkout),
            _PLACEHOLDERS["prompts"]: str(prompts.resolve()),
            _PLACEHOLDERS["output_file"]: str(output_file.resolve()),
            _PLACEHOLDERS["media_dir"]: str(media_dir.resolve()),
        }
        argv = []
        for item in self.argv_template:
            rendered = item
            for placeholder, value in values.items():
                rendered = rendered.replace(placeholder, value)
            argv.append(rendered)
        argv.extend(str(item) for item in activation_args)
        environment = dict(self.env)
        python_root = str(checkout / "python")
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{python_root}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else python_root
        )
        requested_environment = dict(activation_env or {})
        conflicts = sorted(
            name
            for name, value in requested_environment.items()
            if name in environment and environment[name] != value
        )
        if conflicts:
            raise RequestError(
                "candidate activation changes frozen environment: "
                + ", ".join(conflicts)
            )
        environment.update(requested_environment)
        if profile:
            if "--profile" not in argv:
                argv.extend(["--profile", "--num-profiled-timesteps", "5"])
            if profile_dir is None:
                raise RequestError("profile rendering requires profile_dir")
            environment["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"] = str(
                profile_dir.resolve()
            )
        rendered_frozen = {
            name: values.get(value, value) for name, value in self.frozen_flags.items()
        }
        _assert_frozen_flags(
            argv,
            rendered_frozen,
            template=self.argv_template,
        )
        return tuple(argv), environment


def load_launch_request(path: Path) -> LaunchRequest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    request = LaunchRequest.model_validate(data)
    base = path.resolve().parent
    request.sglang_checkout = _resolve_path(request.sglang_checkout, base)
    request.baseline.cwd = _resolve_path(request.baseline.cwd, base)
    request.run_root = _resolve_path(request.run_root, base)
    return request


def normalize_launch_request(
    request: LaunchRequest,
) -> tuple[CampaignGoal, FrozenBenchmarkCommand]:
    checkout = request.sglang_checkout.resolve()
    if not checkout.is_dir():
        raise RequestError(f"SGLang checkout does not exist: {checkout}")
    benchmark_path = checkout / _BENCHMARK_RELATIVE
    if not benchmark_path.is_file():
        raise RequestError(
            f"SGLang native diffusion benchmark is missing: {benchmark_path}"
        )
    cwd = request.baseline.cwd.resolve()
    if not cwd.is_dir():
        raise RequestError(f"baseline working directory does not exist: {cwd}")
    if cwd != checkout:
        raise RequestError(
            "baseline.cwd must be the SGLang checkout root so every detached "
            "worktree preserves command semantics"
        )
    raw_argv, leading_env = _command_argv(request.baseline)
    environment = {**leading_env, **request.baseline.env}
    secret_names = sorted(name for name in environment if _SECRET_NAME.search(name))
    if secret_names:
        raise RequestError(
            "do not persist credentials in the launch request; inherit them "
            "from the remote environment instead: " + ", ".join(secret_names)
        )
    template, flags, mode = _normalize_native_argv(raw_argv, cwd=cwd)
    if flags["--model-path"] != request.model:
        raise RequestError(
            "request model differs from baseline --model-path: "
            f"{request.model!r} != {flags['--model-path']!r}"
        )
    if flags.get("--dataset") != "vbench":
        raise RequestError("Sol-compatible campaigns require --dataset vbench")
    prompt_path = Path(flags["--dataset-path"])
    if not prompt_path.is_absolute():
        prompt_path = (cwd / prompt_path).resolve()
    if not prompt_path.is_file():
        raise RequestError(f"baseline prompt file does not exist: {prompt_path}")
    prompts = [line for line in prompt_path.read_text().splitlines() if line.strip()]
    if "--num-prompts" not in flags:
        raise RequestError("baseline command is missing --num-prompts 5")
    prompt_count = int(flags["--num-prompts"])
    if prompt_count != 5 or len(prompts) < 5:
        raise RequestError(
            "Sol-compatible campaigns require --num-prompts 5 and at least "
            "five non-empty prompts"
        )

    template = _replace_flag_value(template, "--dataset-path", _PLACEHOLDERS["prompts"])
    template = _replace_or_append(
        template, "--output-file", _PLACEHOLDERS["output_file"]
    )
    template = _replace_or_append(template, "--output-path", _PLACEHOLDERS["media_dir"])
    if "--disable-tqdm" not in template:
        template.append("--disable-tqdm")
    original_payload = (
        request.baseline.command
        if request.baseline.command is not None
        else json.dumps(request.baseline.argv, separators=(",", ":"))
    )
    original_digest = hashlib.sha256(original_payload.encode()).hexdigest()
    workload_values = {
        name: str(flags.get(name, default))
        for name, default in {
            "--seed": "42",
            "--width": "32",
            "--height": "32",
            "--num-frames": "1",
            "--fps": "24",
            "--num-inference-steps": "20",
            "--guidance-scale": "7.5",
            "--dtype": "auto",
        }.items()
    }
    frozen_flags = {
        "--model-path": request.model,
        "--dataset": "vbench",
        "--dataset-path": _PLACEHOLDERS["prompts"],
        "--num-prompts": "5",
        **workload_values,
    }
    digest_payload = {
        "argv_template": template,
        "env": environment,
        "frozen_flags": frozen_flags,
        "mode": mode,
    }
    template_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    command = FrozenBenchmarkCommand(
        adapter="sglang_diffusion_offline",
        mode=mode,
        argv_template=template,
        env=environment,
        original_cwd=cwd,
        original_command_sha256=original_digest,
        template_sha256=template_digest,
        frozen_flags=frozen_flags,
    )

    source = request.source or SourceSpec(
        sglang_repo=str(checkout),
        sglang_ref=request.sglang_ref,
    )
    goal = CampaignGoal(
        schema_version=1,
        model=ModelSpec(id=request.model),
        hardware=HardwareSpec(
            environment=request.machine,
            gpu_count=request.gpu_count,
        ),
        workload=WorkloadSpec(
            prompts=prompt_path,
            prompt_count=5,
            seed=int(workload_values["--seed"]),
            height=int(workload_values["--height"]),
            width=int(workload_values["--width"]),
            frames=int(workload_values["--num-frames"]),
            fps=int(workload_values["--fps"]),
            steps=int(workload_values["--num-inference-steps"]),
            guidance=float(workload_values["--guidance-scale"]),
            dtype=workload_values["--dtype"],
            timing_scope=request.baseline.timing_scope,
        ),
        goal=GoalTarget(
            target_speedup=request.target_speedup,
            allow_quality_gated=request.allow_quality_gated,
        ),
        source=source,
        agent=request.agent,
    )
    return goal, command


def _command_argv(
    baseline: BaselineCommandRequest,
) -> tuple[list[str], dict[str, str]]:
    if baseline.argv is not None:
        argv = list(baseline.argv)
        if any(_UNSAFE_TOKEN.search(item) for item in argv):
            raise RequestError("baseline argv contains shell expansion syntax")
    else:
        assert baseline.command is not None
        try:
            punctuation = shlex.shlex(
                baseline.command,
                posix=True,
                punctuation_chars="|&;<>",
            )
            punctuation.whitespace_split = True
            unsafe_shell = any(
                token and all(character in "|&;<>" for character in token)
                for token in punctuation
            )
        except ValueError as error:
            raise RequestError(f"cannot parse baseline command: {error}") from error
        if unsafe_shell or _UNSAFE_TOKEN.search(baseline.command):
            raise RequestError(
                "baseline command contains a shell operator, substitution, or newline"
            )
        try:
            argv = shlex.split(baseline.command, posix=True)
        except ValueError as error:
            raise RequestError(f"cannot parse baseline command: {error}") from error
    environment: dict[str, str] = {}
    while argv and "=" in argv[0] and not argv[0].startswith("-"):
        name, value = argv[0].split("=", 1)
        if not _ENV_NAME.fullmatch(name):
            break
        environment[name] = value
        argv.pop(0)
    if not argv:
        raise RequestError("baseline command contains no executable")
    if any(item in {"|", "||", "&&", ";", ">", ">>", "<"} for item in argv):
        raise RequestError("baseline command requires a shell")
    secret_options = sorted(
        item.split("=", 1)[0]
        for item in argv
        if item.startswith("-") and _SECRET_NAME.search(item.split("=", 1)[0])
    )
    if secret_options:
        raise RequestError(
            "do not put credentials in baseline argv; inherit them from the "
            "remote environment instead: " + ", ".join(secret_options)
        )
    return argv, environment


def _normalize_native_argv(
    argv: list[str], *, cwd: Path
) -> tuple[list[str], dict[str, str], Literal["script", "module"]]:
    template = list(argv)
    mode: Literal["script", "module"]
    script_indexes = [
        index
        for index, value in enumerate(template)
        if Path(value).name == "bench_offline_throughput.py"
    ]
    module_indexes = [
        index
        for index, value in enumerate(template[:-1])
        if value == "-m" and template[index + 1] == _BENCHMARK_MODULE
    ]
    if len(script_indexes) + len(module_indexes) != 1:
        raise RequestError(
            "baseline must invoke SGLang bench_offline_throughput exactly once"
        )
    if script_indexes:
        mode = "script"
        script_index = script_indexes[0]
        supplied = Path(template[script_index])
        supplied = (
            supplied.resolve() if supplied.is_absolute() else (cwd / supplied).resolve()
        )
        expected = (cwd / _BENCHMARK_RELATIVE).resolve()
        if supplied != expected:
            raise RequestError(
                "baseline benchmark script is not from the requested "
                f"SGLang checkout: {supplied} != {expected}"
            )
        template[script_index] = _PLACEHOLDERS["benchmark"]
    else:
        mode = "module"

    flags: dict[str, str] = {}
    index = 0
    while index < len(template):
        token = template[index]
        if token.startswith("--") and "=" in token:
            name, value = token.split("=", 1)
            if name in _VALUE_FLAGS:
                if name in flags:
                    raise RequestError(f"duplicate baseline flag: {name}")
                flags[name] = value
            index += 1
            continue
        if token in _VALUE_FLAGS:
            if token in flags:
                raise RequestError(f"duplicate baseline flag: {token}")
            if index + 1 >= len(template):
                raise RequestError(f"baseline flag has no value: {token}")
            flags[token] = template[index + 1]
            index += 2
            continue
        index += 1
    for required in (
        "--model-path",
        "--dataset",
        "--dataset-path",
        "--num-prompts",
    ):
        if required not in flags:
            raise RequestError(f"baseline command is missing {required}")
    return template, flags, mode


def _replace_flag_value(argv: list[str], flag: str, value: str) -> list[str]:
    result = list(argv)
    found = False
    index = 0
    while index < len(result):
        token = result[index]
        if token == flag:
            if found or index + 1 >= len(result):
                raise RequestError(f"ambiguous baseline flag: {flag}")
            result[index + 1] = value
            found = True
            index += 2
            continue
        if token.startswith(f"{flag}="):
            if found:
                raise RequestError(f"ambiguous baseline flag: {flag}")
            result[index] = f"{flag}={value}"
            found = True
        index += 1
    if not found:
        raise RequestError(f"baseline flag is missing: {flag}")
    return result


def _replace_or_append(argv: list[str], flag: str, value: str) -> list[str]:
    if flag in argv or any(item.startswith(f"{flag}=") for item in argv):
        return _replace_flag_value(argv, flag, value)
    return [*argv, flag, value]


def _flag_values(argv: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and "=" in token:
            name, value = token.split("=", 1)
            values[name] = value
            index += 1
            continue
        if token in _VALUE_FLAGS and index + 1 < len(argv):
            values[token] = argv[index + 1]
            index += 2
            continue
        index += 1
    return values


def _assert_frozen_flags(
    argv: Sequence[str],
    expected: Mapping[str, str],
    *,
    template: Sequence[str],
) -> None:
    values = _flag_values(argv)
    for name, expected_value in expected.items():
        actual = values.get(name)
        original_count = _flag_count(template, name)
        actual_count = _flag_count(argv, name)
        if original_count == 0 and actual_count == 0:
            continue
        if original_count != 1 or actual_count != 1 or actual != expected_value:
            raise RequestError(
                f"rendered command changed frozen flag {name}: "
                f"{actual!r} != {expected_value!r}"
            )


def _flag_count(argv: Sequence[str], name: str) -> int:
    return sum(1 for item in argv if item == name or item.startswith(f"{name}="))


def _resolve_path(path: Path, base: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()
