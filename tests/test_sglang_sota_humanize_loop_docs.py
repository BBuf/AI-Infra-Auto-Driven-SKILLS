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

    def test_humanize_and_kernelpilot_contract_is_explicit(self) -> None:
        text = read_skill_file("SKILL.md")
        template = read_skill_file("references", "refined-plan-template.md")

        self.assertIn("setup-rlcr-loop.sh", text)
        self.assertIn(".humanize/sglang-sota-agent/refined-plan.md", text)
        self.assertIn("KernelPilot Handoff", text)
        self.assertIn("No Simultaneous RLCR Loops", text)
        self.assertIn("analysis/kernelpilot-handoff.md", text)
        self.assertIn("humanize/model-loop-checkpoint.md", text)
        self.assertIn("humanize/sglang-continuation-plan.md", text)
        self.assertIn("Do not rely on resuming the exact same", text)
        self.assertIn("not be live at the same time", text)
        self.assertIn("patching SGLang code, not just benchmark parameters", text)
        self.assertIn("AC-4: Kernel-level bottlenecks use KernelPilot", template)
        self.assertIn("AC-8: Model-loop continuity survives KernelPilot handoff", template)
        self.assertIn("old `.humanize/rlcr/<timestamp>` session", template)
        self.assertIn("setup-rlcr-loop.sh` is launched while the SGLang RLCR loop", template)
        self.assertIn("Eligibility Gate", text)
        self.assertIn("at least `1%` cumulative GPU-time share", text)
        self.assertIn("Cap the KernelPilot Humanize loop at 10 rounds", text)
        self.assertIn("wired into", template)
        self.assertIn("the active model-serving path", template)
        self.assertIn("sub-1% lone", template)
        self.assertIn("AC-7: Stop criteria are satisfied", template)


if __name__ == "__main__":
    unittest.main()
