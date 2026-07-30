from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import redact_argv, redact_environment
from .models import CampaignGoal
from .process import CommandResult, run
from .request import FrozenBenchmarkCommand


_FALLBACK_MARKERS = (
    "falling back to diffusers backend",
    "using diffusers backend",
    "loaded diffusers pipeline",
)


class DriverError(RuntimeError):
    """Raised when a benchmark cannot be used as authoritative evidence."""


@dataclass(frozen=True)
class Activation:
    """A candidate's complete, declared runtime activation."""

    env: Mapping[str, str] = field(default_factory=dict)
    server_args: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class BenchmarkCommand:
    argv: tuple[str, ...]
    env: dict[str, str]
    output_file: Path
    media_dir: Path
    profile_dir: Path | None


@dataclass(frozen=True)
class BenchmarkRun:
    run_dir: Path
    output_file: Path
    normalized_file: Path
    media_dir: Path
    command_receipt: Path
    normalized: dict[str, Any]
    stdout: str
    stderr: str


Runner = Callable[..., CommandResult]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _redact_env(env: Mapping[str, str]) -> dict[str, str]:
    return redact_environment(env)


def _redact_argv(argv: Sequence[str]) -> list[str]:
    return redact_argv(list(argv))


class SGLangDiffusionDriver:
    """Run SGLang's native offline benchmark under a frozen workload."""

    BENCHMARK_RELATIVE = Path(
        "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py"
    )

    def __init__(self, checkout: Path, *, runner: Runner = run) -> None:
        self.checkout = checkout.resolve()
        self.runner = runner
        self.command_template: FrozenBenchmarkCommand | None = None

    @classmethod
    def from_template(
        cls,
        checkout: Path,
        command_template: Path | FrozenBenchmarkCommand,
        *,
        runner: Runner = run,
    ) -> SGLangDiffusionDriver:
        driver = cls(checkout, runner=runner)
        driver.command_template = (
            command_template
            if isinstance(command_template, FrozenBenchmarkCommand)
            else FrozenBenchmarkCommand.model_validate_json(
                command_template.read_text(encoding="utf-8")
            )
        )
        return driver

    @property
    def benchmark_path(self) -> Path:
        benchmark = self.checkout / self.BENCHMARK_RELATIVE
        if not benchmark.is_file():
            raise DriverError(f"SGLang native benchmark is missing: {benchmark}")
        return benchmark

    def build_command(
        self,
        goal: CampaignGoal,
        run_dir: Path,
        *,
        activation: Activation | None = None,
        profile: bool = False,
    ) -> BenchmarkCommand:
        activation = activation or Activation()
        run_dir = run_dir.resolve()
        output_file = run_dir / "outputs" / "benchmark.jsonl"
        media_dir = run_dir / "outputs" / "media"
        prompts = goal.workload.prompts.resolve()
        if not prompts.is_file():
            raise DriverError(f"frozen prompt file is missing: {prompts}")
        if goal.workload.prompt_count != 5:
            raise DriverError("authoritative runs require exactly five prompts")
        if any(not isinstance(value, str) for value in activation.server_args):
            raise TypeError("every activation server argument must be a string")
        if any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in activation.env.items()
        ):
            raise TypeError("activation environment must contain only strings")
        if self.command_template is not None:
            profile_dir = run_dir if profile else None
            argv, environment = self.command_template.render(
                checkout=self.checkout,
                prompts=prompts,
                output_file=output_file,
                media_dir=media_dir,
                activation_env=activation.env,
                activation_args=activation.server_args,
                profile=profile,
                profile_dir=profile_dir,
            )
            return BenchmarkCommand(
                argv=argv,
                env=environment,
                output_file=output_file,
                media_dir=media_dir,
                profile_dir=profile_dir,
            )

        argv = [
            sys.executable,
            str(self.benchmark_path),
            "--model-path",
            goal.model.id,
            "--dataset",
            "vbench",
            "--dataset-path",
            str(prompts),
            "--num-prompts",
            "5",
            "--seed",
            str(goal.workload.seed),
            "--width",
            str(goal.workload.width),
            "--height",
            str(goal.workload.height),
            "--num-frames",
            str(goal.workload.frames),
            "--fps",
            str(goal.workload.fps),
            "--num-inference-steps",
            str(goal.workload.steps),
            "--guidance-scale",
            str(goal.workload.guidance),
            "--dtype",
            goal.workload.dtype,
            "--output-path",
            str(media_dir),
            "--output-file",
            str(output_file),
            "--disable-tqdm",
        ]
        argv.extend(activation.server_args)

        environment = dict(activation.env)
        profile_dir: Path | None = None
        if profile:
            profile_dir = run_dir
            argv.extend(["--profile", "--num-profiled-timesteps", "5"])
            environment["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"] = str(profile_dir)
        return BenchmarkCommand(
            argv=tuple(argv),
            env=environment,
            output_file=output_file,
            media_dir=media_dir,
            profile_dir=profile_dir,
        )

    def run(
        self,
        goal: CampaignGoal,
        run_dir: Path,
        *,
        activation: Activation | None = None,
        profile: bool = False,
    ) -> BenchmarkRun:
        run_dir = run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise DriverError(f"run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)

        command = self.build_command(
            goal, run_dir, activation=activation, profile=profile
        )
        command.output_file.parent.mkdir(parents=True, exist_ok=True)
        command.media_dir.mkdir(parents=True, exist_ok=True)
        receipt = run_dir / "COMMAND.json"
        _atomic_json(
            receipt,
            {
                "schema_version": 1,
                "argv": _redact_argv(command.argv),
                "declared_env": _redact_env(command.env),
                "cwd": str(self.checkout),
                "profile": profile,
                "baseline_command_template_sha256": (
                    self.command_template.template_sha256
                    if self.command_template is not None
                    else None
                ),
            },
        )

        result = self.runner(
            command.argv,
            cwd=self.checkout,
            env=command.env,
            check=False,
        )
        (run_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise DriverError(
                f"SGLang benchmark exited with status {result.returncode}; "
                f"see {run_dir / 'stderr.log'}"
            )
        combined_log = f"{result.stdout}\n{result.stderr}".lower()
        marker = next(
            (value for value in _FALLBACK_MARKERS if value in combined_log), None
        )
        if marker is not None:
            raise DriverError(f"non-native backend evidence rejected: {marker}")

        normalized = self.normalize_output(
            command.output_file,
            timing_scope=goal.workload.timing_scope,
            expected_requests=goal.workload.prompt_count,
        )
        normalized_file = run_dir / "PERFORMANCE.json"
        _atomic_json(normalized_file, normalized)
        return BenchmarkRun(
            run_dir=run_dir,
            output_file=command.output_file,
            normalized_file=normalized_file,
            media_dir=command.media_dir,
            command_receipt=receipt,
            normalized=normalized,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    @staticmethod
    def normalize_output(
        output_file: Path,
        *,
        timing_scope: str,
        expected_requests: int = 5,
    ) -> dict[str, Any]:
        if not output_file.is_file():
            raise DriverError(f"benchmark result is missing: {output_file}")
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            output_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DriverError(
                    f"invalid benchmark JSONL row {line_number}: {error}"
                ) from error
            if isinstance(value, dict):
                rows.append(value)
        if not rows:
            raise DriverError("benchmark result contains no valid object rows")
        raw = rows[-1]
        results = raw.get("results", raw)
        if not isinstance(results, dict):
            raise DriverError("benchmark results must be an object")
        if raw.get("success") is False or results.get("success") is False:
            raise DriverError("benchmark reported an unsuccessful run")

        successful = results.get("successful_requests")
        failed = results.get("failed_requests")
        if successful is not None and int(successful) != expected_requests:
            raise DriverError(
                f"expected {expected_requests} successful requests, got {successful}"
            )
        if failed is not None and int(failed) != 0:
            raise DriverError(f"benchmark reported {failed} failed requests")

        total = SGLangDiffusionDriver._first_number(
            results,
            ("total_duration_seconds", "total_s", "latency"),
        )
        peak = SGLangDiffusionDriver._first_number(
            results,
            ("peak_memory_mb", "peak_memory_mib"),
        )
        if total is None or total <= 0:
            raise DriverError("benchmark latency must be positive")
        if peak is None or peak <= 0:
            raise DriverError("benchmark peak memory must be present and positive")
        return {
            "schema_version": 1,
            "total_s": total,
            "peak_memory_mib": peak,
            "timing_scope": timing_scope,
            "raw_result": raw,
        }

    @staticmethod
    def _first_number(values: Mapping[str, Any], names: Sequence[str]) -> float | None:
        for name in names:
            value = values.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None
