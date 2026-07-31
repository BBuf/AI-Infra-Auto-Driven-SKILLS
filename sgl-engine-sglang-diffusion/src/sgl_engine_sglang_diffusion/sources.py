from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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


@dataclass(frozen=True)
class DerivedSourceSpec:
    """An independently materializable source pinned by a parent gitlink."""

    name: str
    path: str
    repository: str
    commit: str


def derive_submodule_sources(
    manager: SourceManager,
    parent: SourceLock,
    names_by_path: Mapping[str, str],
) -> dict[str, DerivedSourceSpec]:
    """Resolve reviewed submodules from one locked parent without checking them out."""
    cache = manager.checkout_path(parent)
    modules = run(
        ["git", "show", f"{parent.commit}:.gitmodules"],
        cwd=cache,
        check=False,
    )
    if modules.returncode != 0:
        raise RuntimeError(
            f"locked source {parent.name!r} has no readable .gitmodules"
        )
    parser = configparser.ConfigParser()
    try:
        parser.read_string(modules.stdout)
    except configparser.Error as error:
        raise RuntimeError(
            f"locked source {parent.name!r} has invalid .gitmodules: {error}"
        ) from error

    repositories: dict[str, str] = {}
    for section in parser.sections():
        if not section.startswith('submodule "') or not parser.has_option(
            section, "path"
        ):
            continue
        path = parser.get(section, "path").strip()
        repository = parser.get(section, "url", fallback="").strip()
        if path and repository:
            repositories[path] = _normalize_public_git_repository(repository)

    specs: dict[str, DerivedSourceSpec] = {}
    for path, name in names_by_path.items():
        repository = repositories.get(path)
        if not repository:
            raise RuntimeError(
                f"locked source {parent.name!r} does not define submodule {path!r}"
            )
        tree = run(
            ["git", "ls-tree", parent.commit, "--", path],
            cwd=cache,
            check=False,
        )
        match = re.fullmatch(
            rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(path)}\n?",
            tree.stdout,
        )
        if tree.returncode != 0 or match is None:
            raise RuntimeError(
                f"locked source {parent.name!r} has no exact gitlink for {path!r}"
            )
        commit = match.group(1)
        specs[name] = DerivedSourceSpec(
            name=name,
            path=path,
            repository=repository,
            commit=commit,
        )
    return specs


def _normalize_public_git_repository(repository: str) -> str:
    if repository.startswith("git@github.com:"):
        return "https://github.com/" + repository.removeprefix("git@github.com:")
    if repository.startswith("ssh://git@github.com/"):
        return "https://github.com/" + repository.removeprefix(
            "ssh://git@github.com/"
        )
    return repository
