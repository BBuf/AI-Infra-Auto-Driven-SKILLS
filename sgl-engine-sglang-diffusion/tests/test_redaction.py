from __future__ import annotations

from sgl_engine_sglang_diffusion.redaction import redact_argv, redact_environment


def test_redaction_keeps_benchmark_receipts_secret_safe() -> None:
    assert redact_argv(
        ["python", "benchmark.py", "--api-key", "secret", "--mode=fast"]
    ) == ["python", "benchmark.py", "--api-key", "<redacted>", "--mode=fast"]
    assert redact_environment({"CUDA_VISIBLE_DEVICES": "0", "HF_TOKEN": "secret"}) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_TOKEN": "<redacted>",
    }
