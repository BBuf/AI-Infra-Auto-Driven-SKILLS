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

    def test_profiler_gate_lives_inside_rlcr_and_requires_three_tables(self) -> None:
        text = read_skill_file("SKILL.md")

        self.assertIn("Inside Each RLCR Round", text)
        self.assertIn("Gap Decision", text)
        self.assertIn("Required Profiling", text)
        self.assertIn("When SGLang is behind", text)
        self.assertIn("Always profile at least the current best framework", text)
        self.assertIn(
            "If both vLLM and TensorRT-LLM are more than `1%` ahead",
            text,
        )
        self.assertIn("kernel table", text)
        self.assertIn("overlap-opportunity table", text)
        self.assertIn("fuse-pattern table", text)
        self.assertIn("Do not patch SGLang until this report exists for the current gap", text)

    def test_model_pr_history_knowledge_gate_is_explicit(self) -> None:
        text = read_skill_file("SKILL.md")
        template = read_skill_file("references", "refined-plan-template.md")

        self.assertIn("Model PR history knowledge", text)
        self.assertIn("../../model-pr-optimization-history/SKILL.md", text)
        self.assertIn("Phase 0.5: Model PR History Knowledge Gate", text)
        self.assertIn("history/model-pr-history-notes.md", text)
        self.assertIn("model-pr-optimization-history", text)
        self.assertIn("Treat these notes as source and PR memory", text)
        self.assertIn("preserve and consult `history/model-pr-history-notes.md`", text)
        self.assertIn(
            "AC-2: Required model PR history evidence exists before Humanize starts",
            template,
        )
        self.assertIn("matching SGLang model history", template)
        self.assertIn("matching vLLM history", template)
        self.assertIn("model-specific source patch", template)
        self.assertIn("<artifact-root>/history/", template)

    def test_humanize_and_ncu_kernel_contract_is_explicit(self) -> None:
        text = read_skill_file("SKILL.md")
        template = read_skill_file("references", "refined-plan-template.md")

        self.assertIn("setup-rlcr-loop.sh", text)
        self.assertIn(".humanize/sglang-sota-agent/refined-plan.md", text)
        self.assertIn("Kernel Evidence Assist", text)
        self.assertIn("ncu-report-skill", text)
        self.assertNotIn("mit-han-lab/ncu-report-skill", text)
        self.assertIn("kernel/ncu-digests/<version>/", text)
        self.assertIn("humanize/model-loop-checkpoint.md", text)
        self.assertIn("any standalone `.humanize/rlcr` session", text)
        self.assertIn("patch SGLang code, not just benchmark parameters", text)
        self.assertIn(
            "AC-6: Kernel-level work uses ncu-report-skill inside the same model loop",
            template,
        )
        self.assertIn("AC-8: Iteration ledgers and single-loop continuity are preserved", template)
        self.assertIn("ncu-report-skill", template)
        self.assertIn("second `.humanize/rlcr` session is launched", template)
        self.assertIn("each digest under `kernel/ncu-digests/<version>/`", template)
        self.assertIn("Eligibility gate", text)
        self.assertIn("at least `1%` cumulative GPU-time share", text)
        self.assertIn("The kernel candidate is patched directly", template)
        self.assertIn("the active model-serving path", template)
        self.assertIn("sub-1% lone", template)
        self.assertIn("AC-9: Stop criteria are satisfied", template)
        self.assertNotIn("KernelPilot", text)
        self.assertNotIn("KernelPilot", template)

    def test_layer_pipeline_is_in_loop_and_capacity_compute_are_absent(self) -> None:
        text = read_skill_file("SKILL.md")
        template = read_skill_file("references", "refined-plan-template.md")
        normalized = " ".join(text.split())
        template_normalized = " ".join(template.split())

        self.assertIn("Layer Pipeline Deep Dive", text)
        self.assertIn("../llm-pipeline-analysis/SKILL.md", text)
        self.assertIn("analysis/layer-pipeline.md", text)
        self.assertIn("Perfetto time range", text)
        self.assertIn(
            "run `llm-pipeline-analysis` inside the loop",
            normalized,
        )
        self.assertIn(
            "`llm-pipeline-analysis` is run when the three profiler tables are too coarse",
            template_normalized,
        )
        self.assertIn("top hot kernels", template_normalized)
        self.assertNotIn("llm-serving-capacity-planner", text)
        self.assertNotIn("model-compute-simulation", text)
        self.assertNotIn("analysis/capacity.md", text)
        self.assertNotIn("analysis/compute-simulation.md", text)
        self.assertNotIn("llm-serving-capacity-planner", template)
        self.assertNotIn("model-compute-simulation", template)

    def test_readme_links_sglang_sota_flowchart_asset(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        asset = ROOT / "docs" / "assets" / "sglang-sota-performance-loop.svg"
        svg = asset.read_text(encoding="utf-8")

        self.assertIn("SGLang SOTA Performance Loop", readme)
        self.assertIn("docs/assets/sglang-sota-performance-loop.svg", readme)
        self.assertIn("<svg", svg)
        self.assertIn("SGLang SOTA Performance Loop", svg)
        self.assertIn("ncu-report-skill", svg)
        self.assertNotIn("KernelPilot", svg)


if __name__ == "__main__":
    unittest.main()
