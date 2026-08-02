from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "sol-engine-sglang-diffusion"


def run(*argv: object, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in argv],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def init_repo(path: Path, files: dict[str, str | bytes]) -> str:
    path.mkdir(parents=True)
    run("git", "init", "-q", path)
    run("git", "config", "user.name", "Skill Test", cwd=path)
    run("git", "config", "user.email", "skill-test@example.com", cwd=path)
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    run("git", "add", "-A", cwd=path)
    run("git", "commit", "-qm", "fixture", cwd=path)
    return run("git", "rev-parse", "HEAD", cwd=path).stdout.strip()


def create_knowledge_sources(tmp_path: Path) -> tuple[Path, Path]:
    kda = tmp_path / "KDA-Pilot"
    init_repo(
        kda,
        {
            "diffusion/docs/diffusion_kernel_rules.md": "fuse qknorm and rope\n",
            "diffusion/kernels/teacache.py": "# cache hypothesis\n",
            "diffusion/kernels/nvfp4.py": "# quantization hypothesis\n",
        },
    )
    init_repo(kda / "external" / "KernelWiki", {"SKILL.md": "kernel evidence\n"})
    init_repo(
        kda / "external" / "ncu-report-skill", {"SKILL.md": "ncu evidence\n"}
    )
    init_repo(
        kda / "external" / "warp-specialization-report-skill",
        {"SKILL.md": "warp evidence\n"},
    )

    sglang = tmp_path / "sglang"
    init_repo(
        sglang,
        {
            "python/sglang/multimodal_gen/runtime/layers/fused_norm.py": "kernel\n",
            "python/sglang/multimodal_gen/configs/sample/teacache.py": "cache\n",
            "python/sglang/multimodal_gen/tools/build_modelopt_nvfp4.py": "quant\n",
            "python/sglang/multimodal_gen/layers/attention/vmoba.py": "sparse\n",
            "python/sglang/multimodal_gen/distributed/ulysses.py": "topology\n",
        },
    )

    return kda, sglang


def build_pack(tmp_path: Path, kda: Path, sglang: Path, name: str) -> Path:
    output = tmp_path / name
    run(
        sys.executable,
        SKILL / "scripts" / "build_knowledge_pack.py",
        "--kda-root",
        kda,
        "--sglang-root",
        sglang,
        "--output-dir",
        output,
    )
    return output


def test_knowledge_pack_is_deterministic_and_conservatively_routed(
    tmp_path: Path,
) -> None:
    kda, sglang = create_knowledge_sources(tmp_path)
    first = build_pack(tmp_path, kda, sglang, "first")
    second = build_pack(tmp_path, kda, sglang, "second")

    first_bytes = (first / "KNOWLEDGE-MANIFEST.json").read_bytes()
    assert first_bytes == (second / "KNOWLEDGE-MANIFEST.json").read_bytes()
    manifest = json.loads(first_bytes)
    assert manifest["acceptance_authority"] == "upstream-sol-engine"
    assert all(len(source["commit"]) == 40 for source in manifest["sources"])

    by_path = {entry["path"]: entry for entry in manifest["entries"]}
    assert by_path["diffusion/docs/diffusion_kernel_rules.md"][
        "eligible_techniques"
    ] == ["kernel"]
    assert by_path["diffusion/kernels/teacache.py"]["eligible_techniques"] == [
        "cache"
    ]
    assert by_path[
        "python/sglang/multimodal_gen/layers/attention/vmoba.py"
    ]["eligible_techniques"] == ["pisa"]
    assert by_path["python/sglang/multimodal_gen/distributed/ulysses.py"][
        "eligible_techniques"
    ] == ["topology"]

    quant = by_path[
        "python/sglang/multimodal_gen/tools/build_modelopt_nvfp4.py"
    ]
    assert quant["eligible_techniques"] == []
    assert quant["status"] == "knowledge_only_outside_current_sol_registry"
    assert manifest["sglang_history"]


def test_executor_injection_is_lane_specific_and_idempotent(tmp_path: Path) -> None:
    kda, sglang = create_knowledge_sources(tmp_path)
    pack = build_pack(tmp_path, kda, sglang, "pack")
    goal = tmp_path / "goal.md"
    goal.write_text("# Frozen Sol seed goal\n", encoding="utf-8")

    command = (
        sys.executable,
        SKILL / "scripts" / "inject_executor_knowledge.py",
        "--manifest",
        pack / "KNOWLEDGE-MANIFEST.json",
        "--technique",
        "kernel",
        "--goal",
        goal,
    )
    run(*command)
    text = goal.read_text(encoding="utf-8")
    assert "sol-engine-sglang-diffusion:kernel:begin" in text
    assert "diffusion_kernel_rules.md" in text
    assert "fused_norm.py" in text
    assert "teacache.py" not in text
    assert "nvfp4" not in text

    repeated = subprocess.run(
        [str(item) for item in command], text=True, capture_output=True
    )
    assert repeated.returncode != 0
    assert goal.read_text(encoding="utf-8") == text


def test_extract_patch_handles_text_binary_addition_and_deletion(
    tmp_path: Path,
) -> None:
    base = tmp_path / "sglang-base"
    base_commit = init_repo(
        base,
        {
            "python/sglang/multimodal_gen/model.py": "before\n",
            "python/sglang/multimodal_gen/delete_me.py": "obsolete\n",
            "python/sglang/multimodal_gen/blob.bin": b"\x00before\xff",
        },
    )
    candidate = tmp_path / "candidate"
    shutil.copytree(base, candidate, ignore=shutil.ignore_patterns(".git"))
    (candidate / "python/sglang/multimodal_gen/model.py").write_text(
        "after\n", encoding="utf-8"
    )
    (candidate / "python/sglang/multimodal_gen/delete_me.py").unlink()
    (candidate / "python/sglang/multimodal_gen/new.py").write_text(
        "new\n", encoding="utf-8"
    )
    (candidate / "python/sglang/multimodal_gen/blob.bin").write_bytes(
        b"\x00after\xfe"
    )
    patch = tmp_path / "sglang.patch"

    result = run(
        sys.executable,
        SKILL / "scripts" / "extract_sglang_patch.py",
        "--base-repo",
        base,
        "--base-commit",
        base_commit,
        "--candidate-tree",
        candidate,
        "--output",
        patch,
    )

    metadata = json.loads(result.stdout)
    assert metadata["base_commit"] == base_commit
    assert metadata["patch_sha256"]
    patch_bytes = patch.read_bytes()
    assert b"new.py" in patch_bytes
    assert b"delete_me.py" in patch_bytes
    assert b"GIT binary patch" in patch_bytes
    run("git", "apply", "--check", patch, cwd=base)


def test_extract_patch_rejects_candidate_git_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base_commit = init_repo(base, {"tracked.py": "base\n"})
    patch = tmp_path / "bad.patch"
    result = subprocess.run(
        [
            sys.executable,
            SKILL / "scripts" / "extract_sglang_patch.py",
            "--base-repo",
            base,
            "--base-commit",
            base_commit,
            "--candidate-tree",
            base,
            "--output",
            patch,
        ],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "must not contain .git" in result.stderr


def test_extract_patch_rejects_campaign_paths(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base_commit = init_repo(
        base, {"python/sglang/multimodal_gen/model.py": "base\n"}
    )
    candidate = tmp_path / "candidate"
    shutil.copytree(base, candidate, ignore=shutil.ignore_patterns(".git"))
    campaign_file = candidate / "orchestration" / "master.md"
    campaign_file.parent.mkdir()
    campaign_file.write_text("must not ship\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            SKILL / "scripts" / "extract_sglang_patch.py",
            "--base-repo",
            base,
            "--base-commit",
            base_commit,
            "--candidate-tree",
            candidate,
            "--output",
            tmp_path / "bad.patch",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "forbidden campaign path" in result.stderr
