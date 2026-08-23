from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "llm-serving-auto-benchmark"


def read_skill_file(*parts: str) -> str:
    return (SKILL_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


class LlmServingDocsTest(unittest.TestCase):
    def test_current_source_heads_match_refresh_evidence(self) -> None:
        skill = read_skill_file("SKILL.md")
        framework_reference = read_skill_file(
            "references", "framework-reference.md"
        )
        heads = [
            "eec794bce0808ae26cc1dcb84a56b65d2df82af5",
            "bbe8b23e1a2b32a96240b27f63255170d09ef144",
            "da38c1d2e0dffd073b7dfb6d69e15ee7b45d84a9",
            "2706143a8669d50a8f56466b9d340b86922b8f2d",
        ]

        for head in heads:
            with self.subTest(head=head):
                self.assertIn(head, framework_reference)
        for head in heads[1:]:
            with self.subTest(skill_head=head):
                self.assertIn(head, skill)

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
        expected_files = {
            "SKILL.md": "--backend pytorch",
            "references/framework-reference.md": "--backend pytorch",
            "references/container-runbook.md": "--backend pytorch",
            "references/example-plan.yaml": "backend: pytorch",
            "configs/cookbook-llm/README.md": "backend: pytorch",
        }

        for rel_path, expected_text in expected_files.items():
            with self.subTest(rel_path=rel_path):
                text = read_skill_file(*rel_path.split("/"))
                self.assertIn(expected_text, text)

        runbook = read_skill_file("references", "container-runbook.md")
        self.assertIn("separate from the server backend pinned above", runbook)
        self.assertIn("--ipc=host", runbook)
        self.assertIn("-e NCCL_IB_DISABLE=1", runbook)

    def test_dataset_accuracy_is_not_in_default_contract(self) -> None:
        expected_files = [
            "SKILL.md",
            "references/example-plan.yaml",
            "references/framework-reference.md",
            "references/container-runbook.md",
            "references/result-schema.md",
            "configs/cookbook-llm/README.md",
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

    def test_cookbook_documents_intentionally_excluded_current_models(self) -> None:
        readme = read_skill_file("configs", "cookbook-llm", "README.md")

        for model in ("Inkling", "Unlimited OCR", "Kimi K3", "DeepSeek V4"):
            self.assertIn(model, readme)
        self.assertIn("not_verified_at_recorded_head", readme)


if __name__ == "__main__":
    unittest.main()
