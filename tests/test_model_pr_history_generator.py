"""Consistency tests for the model PR-history generator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "rebuild_model_pr_history_from_git.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "rebuild_model_pr_history_from_git", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestModelHistoryConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_generator()

    def test_framework_orders_cover_supported_current_models(self):
        self.assertTrue(
            {"hunyuan3-preview", "moss-vl", "qwen36"}
            <= set(self.mod.FRAMEWORK_MODEL_ORDER["sglang"])
        )
        self.assertTrue(
            {"hunyuan3-preview", "qwen36"}
            <= set(self.mod.FRAMEWORK_MODEL_ORDER["vllm"])
        )
        self.assertNotIn("moss-vl", self.mod.FRAMEWORK_MODEL_ORDER["vllm"])

    def test_every_framework_model_has_title_filter_and_subject_hints(self):
        for framework, models in self.mod.FRAMEWORK_MODEL_ORDER.items():
            for model in models:
                self.assertIn(model, self.mod.MODEL_TITLES)
                self.assertIn(model, self.mod.MODEL_FILTERS[framework])
                self.assertIn(model, self.mod.SUBJECT_HINTS)

    def test_sglang_new_model_filters_select_only_the_intended_surfaces(self):
        files = [
            "docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx",
            "python/sglang/multimodal_gen/runtime/models/dits/hunyuan3d.py",
            "python/sglang/srt/models/moss_vl.py",
            "python/sglang/srt/multimodal/processors/moss_vl.py",
            "docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx",
            "test/registered/ascend/accuracy/qwen3_6_27b/test_model.py",
            "python/sglang/srt/models/qwen3.py",
        ]

        hunyuan = self.mod.selected_files("sglang", "hunyuan3-preview", files)
        self.assertEqual(
            hunyuan,
            ["docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx"],
        )

        moss = self.mod.selected_files("sglang", "moss-vl", files)
        self.assertEqual(
            moss,
            [
                "python/sglang/srt/models/moss_vl.py",
                "python/sglang/srt/multimodal/processors/moss_vl.py",
            ],
        )

        qwen36 = self.mod.selected_files("sglang", "qwen36", files)
        self.assertEqual(
            qwen36,
            [
                "docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx",
                "test/registered/ascend/accuracy/qwen3_6_27b/test_model.py",
            ],
        )

    def test_vllm_new_model_filters_do_not_capture_neighboring_families(self):
        files = [
            "vllm/model_executor/models/hy_v3.py",
            "tests/reasoning/test_hy_v3_reasoning_parser.py",
            "vllm/model_executor/models/hunyuan3d.py",
            "tests/lora/test_qwen36_moe_lora.py",
            "tests/models/language/generation/test_qwen3.py",
            "vllm/model_executor/models/moss_vl.py",
        ]

        hunyuan = self.mod.selected_files("vllm", "hunyuan3-preview", files)
        self.assertEqual(
            hunyuan,
            [
                "tests/reasoning/test_hy_v3_reasoning_parser.py",
                "vllm/model_executor/models/hy_v3.py",
            ],
        )

        qwen36 = self.mod.selected_files("vllm", "qwen36", files)
        self.assertEqual(qwen36, ["tests/lora/test_qwen36_moe_lora.py"])

    def test_qwen36_subject_hints_are_specific(self):
        traces = {
            1: self.mod.TraceInfo(subjects={"[Model] Add Qwen3.6 support"}),
            2: self.mod.TraceInfo(subjects={"[Model] Update Qwen3 core"}),
            3: self.mod.TraceInfo(subjects={"[Test] qwen3_6_35b_a3b"}),
            4: self.mod.TraceInfo(
                files={"tests/lora/test_qwen36_moe_lora.py"},
                subjects={"[LoRA] Support 2D and 3D MoE adapters"},
            ),
        }
        self.assertEqual(
            set(self.mod.filter_traces_by_subject("vllm", "qwen36", traces)),
            {1, 3, 4},
        )
        self.assertEqual(
            set(self.mod.filter_traces_by_subject("sglang", "qwen36", traces)),
            {1, 3},
        )


if __name__ == "__main__":
    unittest.main()
