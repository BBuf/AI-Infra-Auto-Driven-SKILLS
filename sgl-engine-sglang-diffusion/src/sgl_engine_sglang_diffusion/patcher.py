from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .models import AgentProfile
from .process import run


_MODEL_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_CONTENT = (
    re.compile(r"/Users/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"\bHF_TOKEN\b"),
)


class PatchError(RuntimeError):
    """Raised when an integrated SGLang tree is not safely packageable."""


@dataclass(frozen=True)
class PatchBundle:
    directory: Path
    patch: Path
    manifest: Path
    checksums: Path
    apply_script: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


class SGLangPathPolicy:
    """Enforce where generated agent kernels and their tests may live."""

    _SHARED_AGENT_FILES = frozenset(
        {
            PurePosixPath("python/sglang/kernels/agent/__init__.py"),
            PurePosixPath("python/sglang/kernels/agent/registry.py"),
            PurePosixPath("python/sglang/kernels/agent/manifest.py"),
            PurePosixPath("python/sglang/kernels/agent/runtime.py"),
            PurePosixPath("python/sglang/kernels/agent/receipt.py"),
        }
    )

    def __init__(self, model_slug: str) -> None:
        if not _MODEL_SLUG.fullmatch(model_slug):
            raise ValueError(f"unsafe model slug: {model_slug!r}")
        self.model_slug = model_slug
        self.allowed_generated_roots = (
            PurePosixPath(f"python/sglang/kernels/agent/diffusion/{model_slug}"),
            PurePosixPath(f"python/sglang/kernels/ops/diffusion/agent/{model_slug}"),
            PurePosixPath(
                f"python/sglang/kernels/jit/csrc/diffusion/agent/{model_slug}"
            ),
            PurePosixPath(
                "python/sglang/kernels/aot/csrc/diffusion/agent/" f"{model_slug}"
            ),
            PurePosixPath(
                "python/sglang/kernels/aot/include/diffusion/agent/" f"{model_slug}"
            ),
            PurePosixPath(
                "python/sglang/kernels/aot/python/sgl_kernel/diffusion/agent/"
                f"{model_slug}"
            ),
            PurePosixPath(f"test/registered/kernels/ops/diffusion/agent/{model_slug}"),
            PurePosixPath(
                f"test/registered/kernels/benchmark/diffusion/agent/{model_slug}"
            ),
        )

    def validate_changed_paths(self, changes: Iterable[tuple[str, str]]) -> None:
        for status, raw_path in changes:
            path = self._safe_path(raw_path)
            if path.parts and path.parts[0] == ".git":
                raise PatchError(f"edits below .git are forbidden: {path}")
            if status.startswith("A") and self._is_kernel_or_registered_test(path):
                if path in self._SHARED_AGENT_FILES:
                    continue
                if not any(
                    self._inside(path, root) for root in self.allowed_generated_roots
                ):
                    raise PatchError(
                        f"generated kernel/test is outside the agent policy: {path}"
                    )

    @staticmethod
    def _safe_path(raw_path: str) -> PurePosixPath:
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise PatchError(f"unsafe changed path: {raw_path!r}")
        return path

    @staticmethod
    def _inside(path: PurePosixPath, root: PurePosixPath) -> bool:
        return path == root or root in path.parents

    @staticmethod
    def _is_kernel_or_registered_test(path: PurePosixPath) -> bool:
        value = path.as_posix()
        return value.startswith("python/sglang/kernels/") or value.startswith(
            "test/registered/kernels/"
        )


class PatchPackager:
    REQUIRED_AGENT_FILES = (
        "python/sglang/kernels/agent/registry.py",
        "python/sglang/kernels/agent/manifest.py",
        "python/sglang/kernels/agent/runtime.py",
        "python/sglang/kernels/agent/receipt.py",
    )

    def __init__(self, integration_worktree: Path, *, base_sha: str) -> None:
        self.worktree = integration_worktree.resolve()
        self.base_sha = base_sha
        if not re.fullmatch(r"[0-9a-f]{40}", base_sha):
            raise ValueError("base_sha must be a full lowercase Git commit")

    def validate(
        self, *, model_slug: str, profile_id: str | None = None
    ) -> AgentProfile:
        self._assert_git_tree()
        changes = self._changed_paths()
        SGLangPathPolicy(model_slug).validate_changed_paths(changes)
        self._scan_forbidden_content(changes)

        for relative in self.REQUIRED_AGENT_FILES:
            self._require_file(relative)
        profile_path = self._require_file(
            f"python/sglang/kernels/agent/diffusion/{model_slug}/manifest.json"
        )
        profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
        self._validate_derived_checkpoint(profile_payload)
        profile = AgentProfile.model_validate_json(
            profile_path.read_text(encoding="utf-8")
        )
        if profile_id is not None and profile.profile_id != profile_id:
            raise PatchError(
                f"profile ID mismatch: {profile.profile_id!r} != {profile_id!r}"
            )
        if profile.sglang_base_sha != self.base_sha:
            raise PatchError(
                "agent profile was not built from the requested SGLang base SHA"
            )
        diff = run(
            ["git", "diff", "--binary", "--full-index", f"{self.base_sha}..HEAD"],
            cwd=self.worktree,
        ).stdout
        for required_literal in ("--agent-optimization", "off", "auto"):
            if required_literal not in diff:
                raise PatchError(
                    f"SGLang diff is missing runtime option literal "
                    f"{required_literal!r}"
                )
        return profile

    def package(
        self,
        output_dir: Path,
        *,
        model_slug: str,
        profile_id: str | None = None,
        evidence: Sequence[Path] = (),
        cpu_validation_commands: Sequence[Sequence[str]] = (),
        gpu_validation_command: Sequence[str] = (),
        clean_room: bool = True,
    ) -> PatchBundle:
        output_dir = output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise PatchError(f"patch output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        profile = self.validate(model_slug=model_slug, profile_id=profile_id)

        patch_text = run(
            ["git", "diff", "--binary", "--full-index", f"{self.base_sha}..HEAD"],
            cwd=self.worktree,
        ).stdout
        patch_path = output_dir / "sglang.patch"
        _atomic_text(patch_path, patch_text)
        if not patch_text.strip():
            raise PatchError("integrated SGLang diff is empty")

        evidence_dir = output_dir / "evidence"
        copied_evidence: list[dict[str, str]] = []
        for source in evidence:
            resolved = source.resolve()
            if not resolved.is_file():
                raise PatchError(f"evidence file is missing: {source}")
            evidence_dir.mkdir(parents=True, exist_ok=True)
            destination = evidence_dir / resolved.name
            if destination.exists():
                raise PatchError(f"duplicate evidence filename: {resolved.name}")
            shutil.copy2(resolved, destination)
            copied_evidence.append(
                {
                    "path": destination.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(destination),
                }
            )

        manifest_payload = {
            "schema_version": 1,
            "base_sha": self.base_sha,
            "head_sha": self._head_sha(),
            "model_slug": model_slug,
            "profile_id": profile.profile_id,
            "profile_sha256": sha256_file(
                self.worktree
                / f"python/sglang/kernels/agent/diffusion/{model_slug}/manifest.json"
            ),
            "patch_sha256": sha256_file(patch_path),
            "evidence": copied_evidence,
            "cpu_validation_commands": [list(argv) for argv in cpu_validation_commands],
            "gpu_validation_command": list(gpu_validation_command),
        }
        manifest_path = output_dir / "manifest.json"
        _atomic_text(
            manifest_path,
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        )
        script_path = output_dir / "apply_and_verify.sh"
        _atomic_text(
            script_path,
            self._apply_script(
                cpu_validation_commands=cpu_validation_commands,
                gpu_validation_command=gpu_validation_command,
            ),
            mode=0o755,
        )
        checksums_path = output_dir / "SHA256SUMS"
        self._write_checksums(output_dir, checksums_path)

        bundle = PatchBundle(
            directory=output_dir,
            patch=patch_path,
            manifest=manifest_path,
            checksums=checksums_path,
            apply_script=script_path,
        )
        if clean_room:
            self.clean_room_verify(
                bundle,
                cpu_validation_commands=cpu_validation_commands,
                source_hashes=profile.source_hashes,
            )
        return bundle

    def clean_room_verify(
        self,
        bundle: PatchBundle,
        *,
        cpu_validation_commands: Sequence[Sequence[str]],
        source_hashes: Mapping[str, str],
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="sgl-engine-clean-room-") as temporary:
            clean = Path(temporary) / "sglang"
            result = run(
                ["git", "worktree", "add", "--detach", str(clean), self.base_sha],
                cwd=self.worktree,
                check=False,
            )
            if result.returncode != 0:
                raise PatchError(
                    f"failed to create clean-room worktree: {result.stderr}"
                )
            try:
                check = run(
                    ["git", "apply", "--check", str(bundle.patch)],
                    cwd=clean,
                    check=False,
                )
                if check.returncode != 0:
                    raise PatchError(
                        f"clean-room git apply check failed: {check.stderr}"
                    )
                applied = run(
                    ["git", "apply", str(bundle.patch)], cwd=clean, check=False
                )
                if applied.returncode != 0:
                    raise PatchError(f"clean-room patch apply failed: {applied.stderr}")
                whitespace = run(["git", "diff", "--check"], cwd=clean, check=False)
                if whitespace.returncode != 0:
                    raise PatchError(
                        f"clean-room whitespace validation failed: {whitespace.stdout}"
                    )
                for argv in cpu_validation_commands:
                    result = run(list(argv), cwd=clean, check=False)
                    if result.returncode != 0:
                        raise PatchError(
                            f"clean-room validation failed for {list(argv)!r}: "
                            f"{result.stderr}"
                        )
                self._compare_source_hashes(clean, source_hashes)
            finally:
                run(
                    ["git", "worktree", "remove", "--force", str(clean)],
                    cwd=self.worktree,
                    check=False,
                )

    def _changed_paths(self) -> list[tuple[str, str]]:
        result = run(
            ["git", "diff", "--name-status", f"{self.base_sha}..HEAD"],
            cwd=self.worktree,
        )
        changes: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            status = fields[0]
            # For a rename/copy the destination is the relevant content path.
            path = fields[-1]
            changes.append((status, path))
        return changes

    def _scan_forbidden_content(self, changes: Iterable[tuple[str, str]]) -> None:
        for status, relative in changes:
            if status.startswith("D"):
                continue
            path = self.worktree / relative
            if not path.is_file():
                continue
            try:
                value = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in _FORBIDDEN_CONTENT:
                if pattern.search(value):
                    raise PatchError(
                        f"forbidden host/credential content in {relative}: "
                        f"{pattern.pattern}"
                    )

    def _require_file(self, relative: str) -> Path:
        path = (self.worktree / relative).resolve()
        try:
            path.relative_to(self.worktree)
        except ValueError as error:
            raise PatchError(f"required path escapes worktree: {relative}") from error
        if not path.is_file():
            raise PatchError(
                f"required runtime profile artifact is missing: {relative}"
            )
        return path

    def _assert_git_tree(self) -> None:
        if not self.worktree.is_dir():
            raise PatchError(f"integration worktree is missing: {self.worktree}")
        head = self._head_sha()
        if head == self.base_sha:
            raise PatchError("integration worktree contains no committed candidate")
        base = run(
            ["git", "merge-base", "--is-ancestor", self.base_sha, head],
            cwd=self.worktree,
            check=False,
        )
        if base.returncode != 0:
            raise PatchError("requested base is not an ancestor of integration HEAD")
        dirty = run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.worktree,
        ).stdout
        if dirty:
            raise PatchError(f"integration worktree must be clean:\n{dirty}")

    def _head_sha(self) -> str:
        return run(["git", "rev-parse", "HEAD"], cwd=self.worktree).stdout.strip()

    @staticmethod
    def _validate_derived_checkpoint(payload: Mapping[str, Any]) -> None:
        checkpoint = payload.get("derived_checkpoint")
        if checkpoint is None:
            return
        if not isinstance(checkpoint, dict):
            raise PatchError("derived_checkpoint must be an object")
        required = ("uri", "revision", "size_bytes", "sha256")
        missing = [
            name
            for name in required
            if name not in checkpoint or checkpoint[name] in ("", None)
        ]
        if missing:
            raise PatchError(
                "derived checkpoint is not immutable; missing " + ", ".join(missing)
            )
        uri = checkpoint["uri"]
        revision = checkpoint["revision"]
        size = checkpoint["size_bytes"]
        digest = checkpoint["sha256"]
        if (
            not isinstance(uri, str)
            or not isinstance(revision, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise PatchError("derived checkpoint immutable metadata is invalid")

    @staticmethod
    def _compare_source_hashes(root: Path, source_hashes: Mapping[str, str]) -> None:
        for relative, expected in source_hashes.items():
            path = root / relative
            # Knowledge/provenance IDs may also be present. Only relative file
            # paths inside the patched checkout participate in this check.
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not path.is_file()
            ):
                continue
            if not _SHA256.fullmatch(expected):
                raise PatchError(f"invalid source hash for {relative}")
            actual = sha256_file(path)
            if actual != expected:
                raise PatchError(
                    f"clean-room source hash mismatch for {relative}: "
                    f"{actual} != {expected}"
                )

    def _apply_script(
        self,
        *,
        cpu_validation_commands: Sequence[Sequence[str]],
        gpu_validation_command: Sequence[str],
    ) -> str:
        cpu_lines = "\n".join(
            shlex.join(list(argv)) for argv in cpu_validation_commands
        )
        gpu_array = " ".join(
            shlex.quote(argument) for argument in gpu_validation_command
        )
        return f"""#!/usr/bin/env bash
set -euo pipefail

expected_base={shlex.quote(self.base_sha)}
actual_base="$(git rev-parse HEAD)"
if [[ "$actual_base" != "$expected_base" ]]; then
  echo "expected SGLang base $expected_base, got $actual_base" >&2
  exit 2
fi

bundle_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
patch_file="$bundle_dir/sglang.patch"
git apply --check "$patch_file"
git apply "$patch_file"
{cpu_lines}

gpu_command=({gpu_array})
if [[ "${{1:-}}" == "--run-gpu-validation" ]]; then
  if [[ "${{#gpu_command[@]}}" -eq 0 ]]; then
    echo "no GPU validation command was packaged" >&2
    exit 3
  fi
  "${{gpu_command[@]}}"
else
  printf 'GPU revalidation command:'
  printf ' %q' "${{gpu_command[@]}}"
  printf '\\n'
fi
"""

    @staticmethod
    def _write_checksums(root: Path, target: Path) -> None:
        files = sorted(
            path for path in root.rglob("*") if path.is_file() and path != target
        )
        lines = [
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in files
        ]
        _atomic_text(target, "\n".join(lines) + "\n")
