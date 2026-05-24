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
        self.assertIn("--yolo --strict-success", text)
        self.assertIn("strict_success: true", text)
        self.assertIn("clean working tree", text)
        self.assertIn(".humanize/sglang-sota-agent/refined-plan.md", text)
        self.assertIn("Kernel Evidence Assist", text)
        self.assertIn("Single-Loop Kernel Workflow", text)
        self.assertIn("kernel/kernelpilot-knowledge-notes.md", text)
        self.assertIn("kernel/ncu-digests/<version>/", text)
        self.assertIn("ncu-report", text)
        self.assertIn("humanize/model-loop-checkpoint.md", text)
        self.assertIn("Do not start a second `setup-rlcr-loop.sh`", text)
        self.assertIn("any standalone `.humanize/rlcr` session", text)
        self.assertIn("knowledge and source-evidence repository", text)
        self.assertIn("patching SGLang code, not just benchmark parameters", text)
        self.assertIn(
            "AC-4: Kernel-level bottlenecks stay inside the model RLCR loop",
            template,
        )
        self.assertIn("AC-8: Single-loop continuity is preserved", template)
        self.assertIn("strict_success: true", template)
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

    def test_optional_analysis_gates_are_conditional(self) -> None:
        text = read_skill_file("SKILL.md")
        template = read_skill_file("references", "refined-plan-template.md")
        normalized = " ".join(text.split())
        template_normalized = " ".join(template.split())

        self.assertIn("Read these only when the optional analysis gates below trigger", text)
        self.assertIn("Optional Analysis Gates", text)
        self.assertIn("Do not run them as a substitute", text)

        self.assertIn("Capacity Gate", text)
        self.assertIn("../llm-serving-capacity-planner/SKILL.md", text)
        self.assertIn("analysis/capacity.md", text)
        self.assertIn("OOM, KV pool exhaustion", text)

        self.assertIn("Layer Pipeline Gate", text)
        self.assertIn("../llm-pipeline-analysis/SKILL.md", text)
        self.assertIn("analysis/layer-pipeline.md", text)
        self.assertIn("Perfetto time range", text)

        self.assertIn("Compute Simulation Gate", text)
        self.assertIn("../model-compute-simulation/SKILL.md", text)
        self.assertIn("analysis/compute-simulation.md", text)
        self.assertIn("shapes, FLOPs, theoretical time, or MFU", text)

        self.assertIn(
            "preserving any optional capacity, layer-pipeline, or compute-simulation reports",
            normalized,
        )
        self.assertIn(
            "optional capacity/layer-pipeline/compute-simulation reports",
            template_normalized,
        )
        self.assertIn("analysis/compute-simulation.md", template_normalized)


if __name__ == "__main__":
    unittest.main()
