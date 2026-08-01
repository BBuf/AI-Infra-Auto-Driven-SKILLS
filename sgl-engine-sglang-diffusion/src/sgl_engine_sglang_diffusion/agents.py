"""Safe process adapter for coding-agent runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping


_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "KEY")
_REDACTED = "<redacted>"


class AgentRunnerError(RuntimeError):
    """An agent process could not be launched or inspected safely."""


@dataclass(frozen=True)
class AgentProcess:
    pid: int
    argv: tuple[str, ...]
    cwd: Path
    prompt: Path
    receipt: Path


@dataclass(frozen=True)
class ProcessStatus:
    pid: int
    alive: bool
    returncode: int | None


def _contains_secret_marker(value: str) -> bool:
    normalized = value.upper().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_MARKERS)


def redact_argv(argv: list[str]) -> list[str]:
    """Redact values attached to credential-looking command-line options."""
    redacted: list[str] = []
    hide_next = False
    for argument in argv:
        if hide_next:
            redacted.append(_REDACTED)
            hide_next = False
            continue

        if "=" in argument:
            key, value = argument.split("=", 1)
            if _contains_secret_marker(key):
                redacted.append(f"{key}={_REDACTED}")
            else:
                redacted.append(argument)
            continue

        redacted.append(argument)
        if argument.startswith("-") and _contains_secret_marker(argument):
            hide_next = True
    return redacted


def redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return receipt-safe environment overrides."""
    return {
        key: _REDACTED if _contains_secret_marker(key) else value
        for key, value in sorted(environment.items())
    }


def is_codex_exec(command: list[str] | tuple[str, ...]) -> bool:
    return (
        len(command) >= 2 and Path(command[0]).name == "codex" and command[1] == "exec"
    )


def build_agent_argv(
    command: list[str] | tuple[str, ...],
    model: str | None,
    prompt: Path,
) -> list[str]:
    argv = [*command]
    if is_codex_exec(command) and "--json" not in argv:
        argv.append("--json")
    if model:
        argv.extend(["--model", model])
    argv.append(str(prompt))
    return argv


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


class AgentRunner:
    """Launch a configured agent command without a shell."""

    def __init__(self, command: list[str], model: str | None = None):
        if not command or any(
            not isinstance(item, str) or not item for item in command
        ):
            raise ValueError("agent command must contain nonempty argv entries")
        self.command = list(command)
        self.model = model
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    def argv(self, prompt: Path) -> list[str]:
        return build_agent_argv(self.command, self.model, prompt)

    def launch(
        self,
        prompt: Path,
        *,
        cwd: Path,
        receipt: Path,
        stdout: Path | None = None,
        stderr: Path | None = None,
        env: Mapping[str, str] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> AgentProcess:
        prompt = prompt.resolve()
        cwd = cwd.resolve()
        receipt = receipt.resolve()
        if not prompt.is_file():
            raise AgentRunnerError(f"agent prompt does not exist: {prompt}")
        if not cwd.is_dir():
            raise AgentRunnerError(f"agent working directory does not exist: {cwd}")
        if receipt.exists() or receipt.is_symlink():
            raise FileExistsError(f"process receipt already exists: {receipt}")

        stdout = (stdout or receipt.with_suffix(".stdout.log")).resolve()
        stderr = (stderr or receipt.with_suffix(".stderr.log")).resolve()
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stderr.parent.mkdir(parents=True, exist_ok=True)
        argv = self.argv(prompt)
        merged_environment = os.environ.copy()
        environment_overrides = dict(env or {})
        merged_environment.update(environment_overrides)

        with stdout.open("ab") as stdout_stream, stderr.open("ab") as stderr_stream:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=merged_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                shell=False,
                start_new_session=True,
            )

        started_at = datetime.now(UTC).isoformat(timespec="microseconds")
        try:
            _write_json_atomic(
                receipt,
                {
                    "schema_version": 1,
                    "pid": process.pid,
                    "argv": redact_argv(argv),
                    "cwd": str(cwd),
                    "started_at": started_at,
                    "prompt": str(prompt),
                    "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
                    "environment_overrides": redact_environment(environment_overrides),
                    "stdout": str(stdout),
                    "stderr": str(stderr),
                    "start_new_session": True,
                    "context": dict(context or {}),
                },
            )
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait()
            raise

        self._processes[process.pid] = process
        return AgentProcess(
            pid=process.pid,
            argv=tuple(argv),
            cwd=cwd,
            prompt=prompt,
            receipt=receipt.resolve(),
        )

    def poll(self, pid: int) -> ProcessStatus:
        process = self._processes.get(pid)
        if process is not None:
            returncode = process.poll()
            return ProcessStatus(
                pid=pid,
                alive=returncode is None,
                returncode=returncode,
            )

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return ProcessStatus(pid=pid, alive=False, returncode=None)
        except PermissionError:
            return ProcessStatus(pid=pid, alive=True, returncode=None)
        return ProcessStatus(pid=pid, alive=True, returncode=None)

    def terminate(self, pid: int, *, timeout_seconds: float = 5.0) -> None:
        process = self._processes.get(pid)
        # A bare PID from a prior controller process can be reused by the OS.
        # Only terminate children launched and tracked by this runner instance.
        if process is None:
            return
        status = self.poll(pid)
        if not status.alive:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def terminate_all(self) -> None:
        for pid in list(self._processes):
            self.terminate(pid)
