from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> CommandResult:
    """Run an argv vector directly, without invoking a shell."""
    if not argv:
        raise ValueError("argv must not be empty")
    if any(not isinstance(argument, str) for argument in argv):
        raise TypeError("every argv element must be a string")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )
    result = CommandResult(
        tuple(argv), completed.returncode, completed.stdout, completed.stderr
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {list(argv)!r}\n"
            f"{completed.stderr}"
        )
    return result
