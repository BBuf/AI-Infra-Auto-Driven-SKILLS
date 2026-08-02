#!/usr/bin/env python3
"""Export and apply-check a full-index binary patch from a SGLang tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SGLANG_SENTINEL = Path("python/sglang/multimodal_gen")
FORBIDDEN_TOP_LEVEL = {
    "candidates",
    "evals",
    "goals",
    "knowledge",
    "models",
    "orchestration",
    "output",
    "runs",
    "search_space",
    "workflow",
}


def run_bytes(*argv: object, cwd: Path | None = None) -> bytes:
    result = subprocess.run(
        [str(item) for item in argv],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        if not detail:
            detail = result.stdout.decode(errors="replace").strip()
        raise SystemExit(f"command failed: {' '.join(map(str, argv))}: {detail}")
    return result.stdout


def remove_contents(worktree: Path) -> None:
    for path in worktree.iterdir():
        if path.name == ".git":
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def copy_candidate(candidate: Path, worktree: Path) -> None:
    for source in candidate.iterdir():
        if source.name == ".git":
            raise SystemExit("candidate tree must not contain .git metadata")
        target = worktree / source.name
        if source.is_symlink():
            link_target = os.readlink(source)
            resolved = (source.parent / link_target).resolve()
            if Path(link_target).is_absolute() or not resolved.is_relative_to(
                candidate
            ):
                raise SystemExit(f"candidate symlink escapes the SGLang tree: {source}")
            os.symlink(link_target, target)
        elif source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target, follow_symlinks=False)


def remove_worktree(base_repo: Path, path: Path) -> None:
    if not path.exists():
        return
    subprocess.run(
        ["git", "-C", str(base_repo), "worktree", "remove", "--force", str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", required=True, type=Path)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--candidate-tree", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def validate_candidate(candidate: Path) -> None:
    git_metadata = [path for path in candidate.rglob(".git")]
    if git_metadata:
        raise SystemExit(
            f"candidate tree must not contain .git metadata: {git_metadata[0]}"
        )
    for path in candidate.rglob("*"):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        resolved = (path.parent / target).resolve()
        if Path(target).is_absolute() or not resolved.is_relative_to(candidate):
            raise SystemExit(f"candidate symlink escapes the SGLang tree: {path}")
    if not (candidate / SGLANG_SENTINEL).is_dir():
        raise SystemExit(
            f"candidate is not an SGLang source tree: missing {SGLANG_SENTINEL}"
        )


def validate_changed_paths(materialized: Path, base_commit: str) -> None:
    raw = run_bytes(
        "git", "diff", "--cached", "--name-only", "-z", base_commit, cwd=materialized
    )
    paths = [item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item]
    forbidden = [
        path
        for path in paths
        if Path(path).parts and Path(path).parts[0] in FORBIDDEN_TOP_LEVEL
    ]
    if forbidden:
        raise SystemExit(f"candidate contains forbidden campaign path: {forbidden[0]}")


def main() -> int:
    args = parse_args()
    base_repo = args.base_repo.resolve()
    candidate = args.candidate_tree.resolve()
    output = args.output.resolve()
    if not base_repo.is_dir():
        raise SystemExit(f"base repository does not exist: {base_repo}")
    if not candidate.is_dir():
        raise SystemExit(f"candidate tree does not exist: {candidate}")
    validate_candidate(candidate)
    if output.exists():
        raise SystemExit(f"refusing to overwrite an existing patch: {output}")

    base_commit = (
        run_bytes("git", "-C", base_repo, "rev-parse", f"{args.base_commit}^{{commit}}")
        .decode()
        .strip()
    )
    sentinel = run_bytes(
        "git", "-C", base_repo, "ls-tree", "-d", base_commit, SGLANG_SENTINEL
    )
    if not sentinel.strip():
        raise SystemExit(
            f"base commit is not an SGLang tree: missing {SGLANG_SENTINEL}"
        )

    with tempfile.TemporaryDirectory(prefix="sglang-patch-") as temporary:
        root = Path(temporary)
        materialized = root / "materialized"
        verification = root / "verification"
        try:
            run_bytes(
                "git",
                "-C",
                base_repo,
                "worktree",
                "add",
                "--detach",
                materialized,
                base_commit,
            )
            remove_contents(materialized)
            copy_candidate(candidate, materialized)
            run_bytes("git", "add", "-A", cwd=materialized)
            validate_changed_paths(materialized, base_commit)
            patch = run_bytes(
                "git",
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                base_commit,
                cwd=materialized,
            )
            if not patch:
                raise SystemExit("integrated candidate produces an empty SGLang patch")
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".sglang-patch-", dir=output.parent, delete=False
            ) as handle:
                staged_output = Path(handle.name)
                handle.write(patch)

            run_bytes(
                "git",
                "-C",
                base_repo,
                "worktree",
                "add",
                "--detach",
                verification,
                base_commit,
            )
            run_bytes(
                "git", "apply", "--check", "--binary", staged_output, cwd=verification
            )
            os.replace(staged_output, output)
        finally:
            if "staged_output" in locals() and staged_output.exists():
                staged_output.unlink()
            remove_worktree(base_repo, verification)
            remove_worktree(base_repo, materialized)

    print(
        json.dumps(
            {
                "base_commit": base_commit,
                "output": str(output),
                "patch_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
