from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "llm-serving-auto-benchmark"


def read_skill_file(*parts: str) -> str:
    return (SKILL_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


class LlmServingDocsTest(unittest.TestCase):
    def test_tensorrt_llm_backend_policy_is_explicit(self) -> None:
        text = read_skill_file("SKILL.md")

        self.assertIn("`trtllm-serve serve --backend pytorch`", text)
        self.assertIn("do not search tensorrt-llm backend", text.lower())
        self.assertIn("reject", text.lower())
        self.assertIn("non-PyTorch TensorRT-LLM server backend", text)

    def test_example_plan_pins_tensorrt_backend_outside_search_space(self) -> None:
        text = read_skill_file("references", "example-plan.yaml")
        frameworks = text.split("frameworks:\n", 1)[1]
        trt_section = frameworks.split("  tensorrt_llm:\n", 1)[1]

        self.assertIn("backend_policy: fixed_pytorch", trt_section)
        self.assertRegex(
            trt_section,
            r"base_server_flags:\n(?:      .+\n)*      backend: pytorch",
        )

        search_space = trt_section.split("    search_space:\n", 1)[1]
        self.assertNotRegex(search_space, re.compile(r"^\s+backend:", re.MULTILINE))
        self.assertIn("Do not add backend choices here", search_space)

    def test_reference_files_keep_server_backend_pinned(self) -> None:
        expected_files = [
            "references/framework-matrix.md",
            "references/parameter-coverage.md",
            "references/container-runbook.md",
            "references/version-notes.md",
        ]

        for rel_path in expected_files:
            with self.subTest(rel_path=rel_path):
                text = read_skill_file(*rel_path.split("/"))
                self.assertIn("--backend pytorch", text)

        runbook = read_skill_file("references", "container-runbook.md")
        self.assertIn("separate from the server backend pinned above", runbook)
        self.assertIn("--ipc=host", runbook)
        self.assertIn("-e NCCL_IB_DISABLE=1", runbook)

    def test_dataset_accuracy_is_not_in_default_contract(self) -> None:
        expected_files = [
            "SKILL.md",
            "agents/openai.yaml",
            "references/example-plan.yaml",
            "references/framework-matrix.md",
            "references/result-schema.md",
            "references/version-notes.md",
        ]

        blocked_terms = [
            "accuracy",
            "Accuracy",
            "mmlu",
            "MMLU",
            "gsm8k",
            "GSM8K",
            "run_eval",
        ]
        for rel_path in expected_files:
            with self.subTest(rel_path=rel_path):
                text = read_skill_file(*rel_path.split("/"))
                for term in blocked_terms:
                    self.assertNotIn(term, text)

    def test_failed_candidate_table_is_explained(self) -> None:
        skill = read_skill_file("SKILL.md")
        schema = read_skill_file("references", "result-schema.md")

        for text in (skill, schema):
            normalized = " ".join(text.split())
            self.assertIn("tried configs", normalized)
            self.assertIn("not selected", normalized)
            self.assertIn("failed", normalized)
            self.assertIn("skipped", normalized)
            self.assertIn("SLA", normalized)


if __name__ == "__main__":
    unittest.main()
