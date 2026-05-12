from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


REPOS = {
    "flashinfer-ai/flashinfer": "https://github.com/flashinfer-ai/flashinfer.git",
    "Dao-AILab/flash-attention": "https://github.com/Dao-AILab/flash-attention.git",
    "Tencent/hpc-ops": "https://github.com/Tencent/hpc-ops.git",
    "deepseek-ai/DeepGEMM": "https://github.com/deepseek-ai/DeepGEMM.git",
}


def run(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed: {' '.join(args)}\n{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout


def repo_head(url: str) -> str:
    out = run(["git", "ls-remote", url, "HEAD"])
    return out.split()[0]


def kernel_like_paths(repo_dir: Path) -> list[str]:
    out = run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repo_dir)
    paths = []
    for path in out.splitlines():
        low = path.lower()
        if (
            "/csrc/" in f"/{low}/"
            or "/cuda/" in f"/{low}/"
            or "/cutlass/" in f"/{low}/"
            or "/cute/" in f"/{low}/"
            or "/triton/" in f"/{low}/"
            or "/kernel/" in f"/{low}/"
            or "/kernels/" in f"/{low}/"
            or "/bench/" in f"/{low}/"
            or "/benchmarks/" in f"/{low}/"
            or low.endswith((".cu", ".cuh", ".ptx"))
        ):
            paths.append(path)
    return paths


def ensure_sparse_clone(repo: str, url: str, root: Path) -> Path:
    target = root / repo.split("/")[-1]
    if target.exists():
        run(["git", "fetch", "--depth", "1", "origin", "main"], cwd=target)
        run(["git", "checkout", "FETCH_HEAD"], cwd=target)
        return target
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            url,
            str(target),
        ]
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clone-root",
        type=Path,
        default=Path(".kernelpilot/kernel_refs"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.clone_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for repo, url in REPOS.items():
        repo_dir = ensure_sparse_clone(repo, url, args.clone_root)
        rows.append(
            {
                "repo": repo,
                "url": url.removesuffix(".git"),
                "head": repo_head(url),
                "sample_kernel_paths": kernel_like_paths(repo_dir)[:200],
            }
        )
    text = json.dumps({"repositories": rows}, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
