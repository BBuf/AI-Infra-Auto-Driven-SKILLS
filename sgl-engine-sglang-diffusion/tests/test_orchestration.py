from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.agents import AgentRunner, build_agent_argv
from sgl_engine_sglang_diffusion.orchestration import (
    ExecutorManager,
    ExecutorPrompt,
    PromptSection,
    UnsafeDelivery,
)
from sgl_engine_sglang_diffusion.sources import SourceManager
from sgl_engine_sglang_diffusion.state import StateStore

pytest_plugins = ("helpers",)


def _fake_agent(tmp_path: Path) -> Path:
    script = tmp_path / "fake_agent.py"
    script.write_text(
        """
import os
import re
import sys
import time
from pathlib import Path

prompt = Path(sys.argv[-1]).read_text(encoding="utf-8")
match = re.search(r"^Required delivery path: (.+)$", prompt, re.MULTILINE)
if match is None:
    raise SystemExit("missing delivery path")
delivery = Path(match.group(1))
delivery.write_text('{"status": "complete"}\\n', encoding="utf-8")
time.sleep(float(os.environ.get("FAKE_AGENT_DELAY", "0.02")))
""".lstrip(),
        encoding="utf-8",
    )
    return script


def _prompt() -> ExecutorPrompt:
    return ExecutorPrompt(
        correctness_contract=PromptSection(
            "Sol correctness", "correctness contract\n", "locked:sol"
        ),
        technique_scope=PromptSection(
            "Kernel scope", "kernel scope\n", "locked:technique/kernel"
        ),
        placement_rules=PromptSection(
            "SGLang placement", "placement rules\n", "generated:sglang"
        ),
        knowledge=(
            PromptSection(
                "KDA index entry",
                "reference only\n",
                "locked:kda/index.json#entry-1",
            ),
        ),
        baseline=PromptSection(
            "Frozen baseline",
            '{"schema_version": 2, "mean_e2e_s": 10.0, '
            '"workload_total_s": 50.0, "request_count": 5}\n',
            "campaign:BASELINE.json",
        ),
        search_state={"round": 1, "frontier": []},
        rejected_signatures=("prior-failure",),
    )


def test_codex_exec_is_noninteractive_and_writable(tmp_path: Path) -> None:
    prompt = tmp_path / "goal.md"
    prompt.write_text("do the task\n")

    argv = build_agent_argv(["codex", "exec"], None, prompt)

    assert argv == [
        "codex",
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        str(prompt),
    ]


def _manager(
    tmp_path: Path,
    fake_git_repo: Path,
    *,
    delay: str = "0.02",
) -> tuple[ExecutorManager, StateStore, object]:
    state = StateStore.open(tmp_path / "state.sqlite", tmp_path / "events.jsonl")
    state.create_campaign("campaign-1")
    sources = SourceManager(tmp_path / "sources")
    lock = sources.lock("sglang", str(fake_git_repo), "main")
    script = _fake_agent(tmp_path)
    runner = AgentRunner(
        [
            sys.executable,
            str(script),
            "--api-key",
            "argument-secret",
        ],
        model="fake-model",
    )
    manager = ExecutorManager(
        tmp_path / "run",
        state=state,
        sources=sources,
        runner=runner,
        lease_ttl_seconds=60,
        agent_environment={
            "FAKE_AGENT_DELAY": delay,
            "HF_TOKEN": "environment-secret",
        },
    )
    assert manager.agent_environment["GIT_AUTHOR_NAME"] == (
        "sgl-diffusion-engine executor"
    )
    assert manager.agent_environment["GIT_AUTHOR_EMAIL"] == (
        "sgl-diffusion-engine@localhost"
    )
    assert manager.agent_environment["GIT_COMMITTER_NAME"] == (
        "sgl-diffusion-engine executor"
    )
    assert manager.agent_environment["GIT_COMMITTER_EMAIL"] == (
        "sgl-diffusion-engine@localhost"
    )
    return manager, state, lock


def test_executor_git_identity_respects_explicit_environment(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    state = StateStore.open(tmp_path / "state.sqlite", tmp_path / "events.jsonl")
    sources = SourceManager(tmp_path / "sources")
    runner = AgentRunner([sys.executable, str(_fake_agent(tmp_path))])
    manager = ExecutorManager(
        tmp_path / "run",
        state=state,
        sources=sources,
        runner=runner,
        agent_environment={
            "GIT_AUTHOR_NAME": "Explicit Author",
            "GIT_AUTHOR_EMAIL": "author@example.test",
            "GIT_COMMITTER_NAME": "Explicit Committer",
            "GIT_COMMITTER_EMAIL": "committer@example.test",
        },
    )
    try:
        assert manager.agent_environment["GIT_AUTHOR_NAME"] == "Explicit Author"
        assert manager.agent_environment["GIT_AUTHOR_EMAIL"] == (
            "author@example.test"
        )
        assert manager.agent_environment["GIT_COMMITTER_NAME"] == (
            "Explicit Committer"
        )
        assert manager.agent_environment["GIT_COMMITTER_EMAIL"] == (
            "committer@example.test"
        )
    finally:
        manager.close()
        state.close()


def _wait_until_stopped(manager: ExecutorManager, handle: object) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = manager.poll(handle)  # type: ignore[arg-type]
        if not status.alive:
            return
        time.sleep(0.01)
    raise AssertionError("fake agent did not exit")


def test_spawn_is_idempotent_and_uses_one_detached_worktree(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager, state, lock = _manager(tmp_path, fake_git_repo)
    try:
        first = manager.spawn(
            campaign_id="campaign-1",
            technique="kernel",
            source_lock=lock,  # type: ignore[arg-type]
            prompt=_prompt(),
            idempotency_key="spawn-kernel-1",
        )
        second = manager.spawn(
            campaign_id="campaign-1",
            technique="kernel",
            source_lock=lock,  # type: ignore[arg-type]
            prompt=_prompt(),
            idempotency_key="spawn-kernel-1",
        )

        assert first == second
        assert first.worktree.is_dir()
        assert first.worktree != fake_git_repo
        assert first.attempt == 1
        assert len(state.events("campaign-1", event_type="executor_spawned")) == 1

        receipt_text = first.receipt.read_text(encoding="utf-8")
        assert "argument-secret" not in receipt_text
        assert "environment-secret" not in receipt_text
        receipt = json.loads(receipt_text)
        assert receipt["argv"][-1] == str(first.prompt)
        assert receipt["start_new_session"] is True
        assert receipt["environment_overrides"]["HF_TOKEN"] == "<redacted>"
    finally:
        manager.close()
        state.close()


def test_poll_separates_liveness_from_delivery(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager, state, lock = _manager(tmp_path, fake_git_repo, delay="0.5")
    try:
        handle = manager.spawn(
            campaign_id="campaign-1",
            technique="kernel",
            source_lock=lock,  # type: ignore[arg-type]
            prompt=_prompt(),
            idempotency_key="spawn-kernel-1",
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = manager.poll(handle)
            if status.delivered:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("fake delivery was not observed")

        assert status.delivered is True
        assert status.delivery == handle.delivery
        assert status.alive is True
    finally:
        manager.close()
        state.close()


def test_spawn_recovers_receipted_process_without_duplicate_launch(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager, state, lock = _manager(tmp_path, fake_git_repo, delay="0.5")
    try:
        first = manager.spawn(
            campaign_id="campaign-1",
            technique="kernel",
            source_lock=lock,  # type: ignore[arg-type]
            prompt=_prompt(),
            idempotency_key="spawn-kernel-1",
        )
        (first.root / "attempt-001.json").unlink()
        (first.root / "executor.json").unlink()

        recovered = manager.spawn(
            campaign_id="campaign-1",
            technique="kernel",
            source_lock=lock,  # type: ignore[arg-type]
            prompt=_prompt(),
            idempotency_key="spawn-kernel-1",
        )

        assert recovered.pid == first.pid
        assert recovered.receipt == first.receipt
        assert (first.root / "attempt-001.json").is_file()
        assert len(state.events("campaign-1", event_type="executor_spawned")) == 1
    finally:
        manager.close()
        state.close()


def test_resume_appends_exact_feedback_and_reuses_worktree(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager, state, lock = _manager(tmp_path, fake_git_repo)
    try:
        first = manager.spawn(
            campaign_id="campaign-1",
            technique="kernel",
            source_lock=lock,  # type: ignore[arg-type]
            prompt=_prompt(),
            idempotency_key="spawn-kernel-1",
        )
        _wait_until_stopped(manager, first)
        feedback = "Measured speedup is inconsistent.\nRecompute from benchmark.json."
        resumed = manager.resume(
            first,
            feedback=feedback,
            idempotency_key="resume-kernel-1",
        )
        repeated = manager.resume(
            first,
            feedback=feedback,
            idempotency_key="resume-kernel-1",
        )

        assert resumed == repeated
        assert resumed.worktree == first.worktree
        assert resumed.attempt == 2
        assert resumed.pid != first.pid
        assert (first.root / "feedback-001.md").read_text() == feedback
        assert feedback in first.prompt.read_text(encoding="utf-8")
        assert (first.root / "rejected-delivery-001.json").is_file()
        assert len(state.events("campaign-1", event_type="executor_resumed")) == 1
    finally:
        manager.close()
        state.close()


def test_delivery_symlink_outside_worktree_is_rejected(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager, state, lock = _manager(tmp_path, fake_git_repo, delay="1")
    try:
        handle = manager.spawn(
            campaign_id="campaign-1",
            technique="kernel",
            source_lock=lock,  # type: ignore[arg-type]
            prompt=_prompt(),
            idempotency_key="spawn-kernel-1",
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not handle.delivery.exists():
            time.sleep(0.01)
        manager.runner.terminate(handle.pid)
        handle.delivery.unlink()
        outside = tmp_path / "outside-delivery.json"
        outside.write_text("{}\n", encoding="utf-8")
        handle.delivery.symlink_to(outside)

        with pytest.raises(UnsafeDelivery, match="escapes|symbolic link"):
            manager.poll(handle)
    finally:
        manager.close()
        state.close()


def test_prompt_has_ordered_precedence_and_hashes(tmp_path: Path) -> None:
    # Prompt assembly is exercised through spawn elsewhere. This assertion keeps
    # the security-relevant precedence visible and every inserted section hashed.
    from sgl_engine_sglang_diffusion.orchestration import assemble_executor_prompt

    worktree = tmp_path / "executor" / "worktree"
    worktree.mkdir(parents=True)
    text = assemble_executor_prompt(
        _prompt(),
        worktree=worktree,
        delivery=worktree / "DELIVERY.json",
    )
    positions = [
        text.index("1. Sol correctness"),
        text.index("2. Kernel scope"),
        text.index("3. SGLang placement"),
        text.index("4.1. KDA index entry"),
        text.index("5. Frozen baseline"),
        text.index("6. Current search state"),
        text.index("7. Assigned worktree"),
    ]
    assert positions == sorted(positions)
    assert text.count("SHA-256:") == 7
    assert "cannot override" in text
