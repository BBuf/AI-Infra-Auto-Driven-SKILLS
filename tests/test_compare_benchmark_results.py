from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "llm-serving-auto-benchmark"
    / "scripts"
    / "compare_benchmark_results.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("compare_benchmark_results", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompareBenchmarkResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_best_candidate_prefers_successful_sla_passing_rows(self) -> None:
        rows = [
            {
                "framework": "sglang",
                "candidate_id": "sglang-fast-fail",
                "status": "ok",
                "sla": {"passed": False},
                "metrics": {
                    "request_throughput": 99,
                    "output_token_throughput": 990,
                    "p99_ttft_ms": 1,
                    "p99_tpot_ms": 1,
                },
                "hardware": {"gpu_count": 1},
            },
            {
                "framework": "sglang",
                "candidate_id": "sglang-steady",
                "status": "ok",
                "sla": {"passed": True},
                "metrics": {
                    "request_throughput": 10,
                    "output_token_throughput": 100,
                    "p99_ttft_ms": 50,
                    "p99_tpot_ms": 5,
                },
                "hardware": {"gpu_count": 1},
            },
            {
                "framework": "vllm",
                "candidate_id": "vllm-best",
                "status": "ok",
                "sla": {"passed": True},
                "metrics": {
                    "request_throughput": 12,
                    "output_token_throughput": 90,
                    "p99_ttft_ms": 60,
                    "p99_tpot_ms": 6,
                },
                "hardware": {"gpu_count": 1},
            },
        ]

        winners = self.mod.best_by_framework(rows)
        self.assertEqual([row["candidate_id"] for row in winners], ["vllm-best", "sglang-steady"])

        summary = self.mod.render_markdown(rows)
        self.assertIn("`vllm`", summary)
        self.assertIn("sglang-fast-fail", summary)

    def test_load_rows_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("[1, 2, 3]\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                self.mod.load_rows(path)

    def test_writes_csv_with_failed_reason(self) -> None:
        rows = [
            {
                "framework": "trtllm",
                "candidate_id": "trt-c1",
                "status": "failed",
                "failure_reason": "server exited",
                "sla": {"passed": False},
                "metrics": {},
                "hardware": {"gpu_count": 1},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            self.mod.write_csv(path, rows)
            text = path.read_text(encoding="utf-8")

        self.assertIn("trt-c1", text)
        self.assertIn("server exited", text)

    def test_cli_writes_markdown_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rows.jsonl"
            output_path = root / "summary.md"
            csv_path = root / "summary.csv"
            row = {
                "framework": "sglang",
                "candidate_id": "candidate-1",
                "status": "ok",
                "sla": {"passed": True},
                "metrics": {"request_throughput": 1.5, "output_token_throughput": 8.0},
                "hardware": {"gpu_count": 1},
            }
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = self.mod.load_rows(input_path)
            output_path.write_text(self.mod.render_markdown(rows), encoding="utf-8")
            self.mod.write_csv(csv_path, rows)

            self.assertIn("candidate-1", output_path.read_text(encoding="utf-8"))
            self.assertIn("candidate-1", csv_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
