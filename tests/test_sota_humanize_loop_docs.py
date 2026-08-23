from __future__ import annotations

import unittest
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOTA_SKILL_ROOTS = [
    ROOT / "skills" / "sglang-sota-humanize-loop",
    ROOT / "skills" / "vllm-sota-humanize-loop",
]
REMOVED_STRICT_FLAG = "--strict" "-success"
REMOVED_STRICT_STATE = "strict" "_success"
PROMPT_PAIRS = [
    (
        ROOT / "prompts" / "sglang-sota-b200-prompts.md",
        ROOT / "prompts" / "sglang-sota-b200-codex-goal-prompts.md",
    ),
    (
        ROOT / "prompts" / "sglang-sota-h200-prompts.md",
        ROOT / "prompts" / "sglang-sota-h200-codex-goal-prompts.md",
    ),
]
SOURCE_HEADS = {
    "sglang": "eec794bce0808ae26cc1dcb84a56b65d2df82af5",
    "vllm": "bbe8b23e1a2b32a96240b27f63255170d09ef144",
    "tensorrt_llm": "da38c1d2e0dffd073b7dfb6d69e15ee7b45d84a9",
    "tokenspeed": "2706143a8669d50a8f56466b9d340b86922b8f2d",
}


def _model_metadata(text: str) -> list[tuple[str, ...]]:
    fields = [
        "model_id",
        "target_hardware",
        "minimum_gpu_count",
        "precision_quantization",
        "initial_deployment",
    ]
    blocks = re.findall(r"```text\n(.*?)\n```", text, re.S)
    metadata = []
    for block in blocks:
        if not re.search(r"^model_id:", block, re.M):
            continue
        values = []
        for field in fields:
            match = re.search(rf"^{field}: (.+)$", block, re.M)
            if not match:
                raise AssertionError(f"missing {field} in prompt block")
            values.append(match.group(1))
        metadata.append(tuple(values))
    return metadata


class SotaHumanizeLoopDocsTest(unittest.TestCase):
    def test_prompt_variants_share_ordered_model_metadata(self) -> None:
        for regular_path, goal_path in PROMPT_PAIRS:
            with self.subTest(hardware=regular_path.name):
                regular = regular_path.read_text(encoding="utf-8")
                goal = goal_path.read_text(encoding="utf-8")
                self.assertEqual(_model_metadata(regular), _model_metadata(goal))

    def test_b200_prompts_use_verified_minimax_m3_and_qwen36_shapes(self) -> None:
        for prompt_path in PROMPT_PAIRS[0]:
            with self.subTest(prompt=prompt_path.name):
                text = prompt_path.read_text(encoding="utf-8")
                self.assertIn("MiniMaxAI/MiniMax-M3-MXFP8", text)
                self.assertIn("Qwen/Qwen3.6-35B-A3B-FP8", text)
                self.assertNotIn("MiniMaxAI/MiniMax-M2.7", text)
                self.assertRegex(
                    text,
                    r"(?s)model_id: Qwen/Qwen3\.6-35B-A3B-FP8.*?"
                    r"target_hardware: single-node 1x NVIDIA B200",
                )
                self.assertRegex(
                    text,
                    r"(?s)model_id: MiniMaxAI/MiniMax-M3-MXFP8.*?"
                    r"target_hardware: single-node 8x NVIDIA B200",
                )

    def test_skills_record_current_immutable_source_heads(self) -> None:
        sglang = (SOTA_SKILL_ROOTS[0] / "SKILL.md").read_text(encoding="utf-8")
        vllm = (SOTA_SKILL_ROOTS[1] / "SKILL.md").read_text(encoding="utf-8")
        for framework, source_head in SOURCE_HEADS.items():
            with self.subTest(framework=framework):
                self.assertIn(source_head, sglang)
                if framework != "tokenspeed":
                    self.assertIn(source_head, vllm)

    def test_templates_share_framework_neutral_checkpoint_evidence(self) -> None:
        templates = [
            (root / "references" / "refined-plan-template.md").read_text(
                encoding="utf-8"
            )
            for root in SOTA_SKILL_ROOTS
        ]
        shared_fields = [
            "immutable source heads",
            "PR state",
            "validation evidence",
            "known limitations",
            "selected comparison frameworks",
            "user-excluded or unsupported frameworks",
            "current leading selected comparison result",
            "remaining gap",
        ]
        for field in shared_fields:
            with self.subTest(field=field):
                for template in templates:
                    self.assertIn(field, template)

    def test_rlcr_startup_uses_supported_humanize_options(self) -> None:
        for skill_root in SOTA_SKILL_ROOTS:
            with self.subTest(skill=skill_root.name):
                skill = (skill_root / "SKILL.md").read_text(encoding="utf-8")

                self.assertIn("setup-rlcr-loop.sh", skill)
                self.assertIn("--yolo", skill)
                self.assertNotIn(REMOVED_STRICT_FLAG, skill)
                self.assertNotIn(REMOVED_STRICT_STATE, skill)

    def test_refined_plan_templates_do_not_require_removed_strict_state(self) -> None:
        for skill_root in SOTA_SKILL_ROOTS:
            with self.subTest(skill=skill_root.name):
                template = (
                    skill_root / "references" / "refined-plan-template.md"
                ).read_text(encoding="utf-8")

                self.assertNotIn(REMOVED_STRICT_FLAG, template)
                self.assertNotIn(REMOVED_STRICT_STATE, template)
                self.assertIn("current_round: 0", template)
                self.assertIn("ask_codex_question: false", template)
                self.assertIn("round-0-prompt.md", template)


if __name__ == "__main__":
    unittest.main()
