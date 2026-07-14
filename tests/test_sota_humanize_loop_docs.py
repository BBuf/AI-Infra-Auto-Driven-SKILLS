from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOTA_SKILL_ROOTS = [
    ROOT / "skills" / "sglang-sota-humanize-loop",
    ROOT / "skills" / "vllm-sota-humanize-loop",
]
REMOVED_STRICT_FLAG = "--strict" "-success"
REMOVED_STRICT_STATE = "strict" "_success"


class SotaHumanizeLoopDocsTest(unittest.TestCase):
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
