from __future__ import annotations

import re
from pathlib import Path

from .models import SourceLock
from .process import run


_SAFE_SOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SourceManager:
    """Manage shared bare Git caches and immutable detached worktrees."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def lock(self, name: str, repository: str, requested_ref: str) -> SourceLock:
        self._validate_source_name(name)
        if not repository:
            raise ValueError("repository must not be empty")
        if not requested_ref:
            raise ValueError("requested_ref must not be empty")

        cache = self._cache_path(name)
        if cache.exists():
            if not cache.is_dir():
                raise RuntimeError(f"source cache is not a directory: {cache}")
            self._assert_bare_cache(cache)
            configured_repository = run(
                ["git", "remote", "get-url", "origin"], cwd=cache
            ).stdout.strip()
            if not self._same_repository(configured_repository, repository):
                raise RuntimeError(
                    f"source name {name!r} is already bound to "
                    f"{configured_repository!r}, not {repository!r}"
                )
        else:
            run(
                [
                    "git",
                    "clone",
                    "--bare",
                    "--filter=blob:none",
                    repository,
                    str(cache),
                ],
                cwd=self.root,
            )
            self._assert_bare_cache(cache)

        run(
            ["git", "fetch", "--force", "--no-tags", "origin", requested_ref],
            cwd=cache,
        )
        commit = run(
            ["git", "rev-parse", "--verify", "FETCH_HEAD^{commit}"], cwd=cache
        ).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError(
                f"Git resolved {requested_ref!r} to an invalid commit: {commit!r}"
            )
        # FETCH_HEAD is overwritten by the next fetch. Pin every lock in a
        # private ref namespace so an older campaign remains materializable
        # even after later locks and Git garbage collection.
        run(
            ["git", "update-ref", f"refs/sgl-engine-locks/{commit}", commit],
            cwd=cache,
        )
        return SourceLock(
            name=name,
            repository=repository,
            requested_ref=requested_ref,
            commit=commit,
        )

    def checkout_path(self, lock: SourceLock) -> Path:
        """Return the shared bare cache that contains a locked commit."""
        self._validate_source_name(lock.name)
        cache = self._cache_path(lock.name)
        self._assert_bare_cache(cache)
        result = run(
            ["git", "cat-file", "-e", f"{lock.commit}^{{commit}}"],
            cwd=cache,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"locked commit {lock.commit} is missing from source cache {cache}"
            )
        return cache

    def create_worktree(self, lock: SourceLock, destination: Path) -> Path:
        cache = self.checkout_path(lock)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"worktree destination already exists: {destination}")

        resolved_destination = destination.resolve()
        if (
            resolved_destination == cache
            or cache in resolved_destination.parents
            or resolved_destination == self.root
        ):
            raise ValueError(
                f"worktree destination must be outside the shared cache: {destination}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                str(resolved_destination),
                lock.commit,
            ],
            cwd=cache,
        )
        self.assert_clean_worktree(resolved_destination)
        return resolved_destination

    def assert_clean_worktree(self, worktree: Path) -> None:
        status = run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=worktree,
        ).stdout
        if status:
            raise RuntimeError(f"worktree is dirty: {worktree}\n{status}")

    def _cache_path(self, name: str) -> Path:
        path = (self.root / f"{name}.git").resolve()
        if path.parent != self.root:
            raise ValueError(f"unsafe source cache name: {name!r}")
        return path

    @staticmethod
    def _validate_source_name(name: str) -> None:
        if not _SAFE_SOURCE_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError(f"unsafe source name: {name!r}")

    @staticmethod
    def _assert_bare_cache(cache: Path) -> None:
        if not cache.is_dir():
            raise RuntimeError(f"source cache does not exist: {cache}")
        result = run(
            ["git", "rev-parse", "--is-bare-repository"], cwd=cache
        ).stdout.strip()
        if result != "true":
            raise RuntimeError(f"source cache is not a bare Git repository: {cache}")

    @staticmethod
    def _same_repository(left: str, right: str) -> bool:
        def normalize(value: str) -> str:
            path = Path(value).expanduser()
            if path.exists():
                return str(path.resolve())
            return value.removesuffix("/")

        return normalize(left) == normalize(right)
