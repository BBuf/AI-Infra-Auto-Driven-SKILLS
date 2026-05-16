from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "sglang-sota-humanize-loop"


def read_skill_file(*parts: str) -> str:
    return SKILL_ROOT.joinpath(*parts).read_text(encoding="utf-8")


class SglangSotaHumanizeLoopDocsTest(unittest.TestCase):
    def test_fixed_benchmark_gate_preserves_fair_default_workload(self) -> None:
        text = read_skill_file("SKILL.md")

        self.assertIn("Fixed Fair Benchmark Gate", text)
        self.assertIn("SGLang, vLLM, and TensorRT-LLM", text)
        self.assertIn("`num_prompts: 80`", text)
        self.assertIn("random input `1000`, output `1000`", text)
        self.assertIn("random input `8000`, output `1000`", text)
        self.assertIn("not a cartesian", text)
        self.assertIn("Do not replace those scenarios with an easier smoke dataset", text)
        self.assertIn("`trtllm-serve serve --backend pytorch`", text)

    def test_profiler_gate_requires_competitor_profiles_and_three_tables(self) -> None:
        text = read_skill_file("SKILL.md")

        self.assertIn("Always profile at least the current best framework", text)
        self.assertIn(
            "If both vLLM and TensorRT-LLM are more than `1%` ahead",
            text,
        )
        self.assertIn("kernel table", text)
        self.assertIn("overlap-opportunity table", text)
        self.assertIn("fuse-pattern table", text)
        self.assertIn("Do not patch SGLang until this report exists", text)

    def test_model_pr_history_knowledge_gate_is_explicit(self) -> None:
        text = read_skill_file("SKILL.md")
        template = read_skill_file("references", "refined-plan-template.md")

        self.assertIn("Model PR history knowledge", text)
        self.assertIn("../../model-pr-optimization-history/SKILL.md", text)
        self.assertIn("Phase 0.5: Model PR History Knowledge Gate", text)
        self.assertIn("history/model-pr-history-notes.md", text)
        self.assertIn("model-pr-optimization-history", text)
        self.assertIn("Treat these notes like KernelPilot knowledge", text)
        self.assertIn("preserving and consulting `history/model-pr-history-notes.md`", text)
        self.assertIn(
            "AC-2: Required model PR history and profiler evidence exists before patching",
            template,
        )
        self.assertIn("matching SGLang model history", template)
        self.assertIn("matching vLLM history", template)
        self.assertIn("model-specific source patch", template)
        self.assertIn("<artifact-root>/history/", template)

    def test_humanize_and_kernel_assist_contract_is_explicit(self) -> None:
        text = read_skill_file("SKILL.md")
        template = read_skill_file("references", "refined-plan-template.md")

        self.assertIn("setup-rlcr-loop.sh", text)
        self.assertIn(".humanize/sglang-sota-agent/refined-plan.md", text)
        self.assertIn("Kernel Evidence Assist", text)
        self.assertIn("Single-Loop Kernel Workflow", text)
        self.assertIn("kernel/kernelpilot-knowledge-notes.md", text)
        self.assertIn("kernel/ncu-digests/<version>/", text)
        self.assertIn("ncu-report", text)
        self.assertIn("humanize/model-loop-checkpoint.md", text)
        self.assertIn("Do not start KernelPilot's `setup-rlcr-loop.sh`", text)
        self.assertIn("any standalone `.humanize/rlcr` session", text)
        self.assertIn("knowledge and source-evidence repository", text)
        self.assertIn("patching SGLang code, not just benchmark parameters", text)
        self.assertIn(
            "AC-4: Kernel-level bottlenecks stay inside the model RLCR loop",
            template,
        )
        self.assertIn("AC-8: Single-loop continuity is preserved", template)
        self.assertIn("KernelPilot is used only as a knowledge/source-evidence", template)
        self.assertIn("second `.humanize/rlcr` session is launched", template)
        self.assertIn("each digest under `kernel/ncu-digests/<version>/`", template)
        self.assertIn("Eligibility Gate", text)
        self.assertIn("at least `1%` cumulative GPU-time share", text)
        self.assertIn("The kernel candidate is patched directly", template)
        self.assertIn("the active model-serving path", template)
        self.assertIn("sub-1% lone", template)
        self.assertIn("AC-7: Stop criteria are satisfied", template)
        self.assertNotIn("KernelPilot Handoff", text)
        self.assertNotIn("sglang-continuation-plan.md", text)
        self.assertNotIn("Cap the KernelPilot Humanize loop", text)


if __name__ == "__main__":
    unittest.main()
