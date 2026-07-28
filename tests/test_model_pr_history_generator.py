"""Consistency tests for the model PR-history generator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


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

    def test_transient_fetch_errors_do_not_poison_the_cache(self):
        key = "vllm-project/vllm#123"
        cache = {
            "prs": {
                key: {
                    "info": {
                        "fetch_error": "API rate limit exceeded (HTTP 403)",
                    },
                    "files": [],
                }
            }
        }
        info = {
            "number": 123,
            "title": "real PR",
            "html_url": "https://github.com/vllm-project/vllm/pull/123",
        }
        files = [{"filename": "tests/lora/test_qwen36_moe_lora.py"}]

        with mock.patch.object(self.mod, "gh_api", side_effect=[info, files]):
            fetched_info, fetched_files = self.mod.fetch_pr_bundle(
                "vllm", 123, cache
            )

        self.assertEqual(fetched_info, info)
        self.assertEqual(fetched_files, files)
        self.assertEqual(cache["prs"][key], {"info": info, "files": files})

    def test_empty_cached_success_is_refetched(self):
        key = "sgl-project/sglang#28940"
        cache = {"prs": {key: {"info": {}, "files": []}}}
        info = {
            "number": 28940,
            "title": "MOSS-VL preprocessing optimizations",
            "html_url": "https://github.com/sgl-project/sglang/pull/28940",
        }
        files = [{"filename": "python/sglang/srt/models/moss_vl.py"}]

        with mock.patch.object(self.mod, "gh_api", side_effect=[info, files]):
            fetched_info, fetched_files = self.mod.fetch_pr_bundle(
                "sglang", 28940, cache
            )

        self.assertEqual(fetched_info, info)
        self.assertEqual(fetched_files, files)
        self.assertEqual(cache["prs"][key], {"info": info, "files": files})

    def test_existing_prs_are_preserved_from_current_head(self):
        history = (
            "https://github.com/sgl-project/sglang/pull/100\n"
            "https://github.com/sgl-project/sglang/pull/101\n"
        )

        def fake_run(command, *_args, **_kwargs):
            if command[:3] == ["git", "merge-base", "HEAD"]:
                return "base-sha\n"
            return history

        with mock.patch.object(self.mod, "run", side_effect=fake_run) as run:
            numbers = self.mod.extract_existing_prs("sglang", "kimi")

        self.assertEqual(numbers, {100, 101})
        show_refs = [
            call.args[0][2]
            for call in run.call_args_list
            if call.args[0][:2] == ["git", "show"]
        ]
        self.assertTrue(any(ref.startswith("HEAD:") for ref in show_refs))
        self.assertTrue(any(ref.startswith("base-sha:") for ref in show_refs))

    def test_existing_cards_survive_unavailable_github_metadata(self):
        card_en = """\
### PR #36127 - Add Kimi Audio

- Link: https://github.com/vllm-project/vllm/pull/36127
- Status/date: merged / 2026-03-11
- Trace source: immutable commit evidence
- Key implementation: preserved implementation details.
"""
        card_zh = """\
### PR #36127 - 支持 Kimi Audio

- 链接: https://github.com/vllm-project/vllm/pull/36127
- 状态/时间: merged / 2026-03-11
- 反查来源: 不可变提交证据
- 实现要点: 保留实现细节。
"""
        failed = self.mod.PRBundle(
            framework="vllm",
            repo="vllm-project/vllm",
            number=36127,
            info={
                "number": 36127,
                "title": "unavailable PR #36127",
                "html_url": "https://github.com/vllm-project/vllm/pull/36127",
                "state": "unknown",
                "fetch_error": "gh api failed (HTTP 404)",
            },
            files=[],
            trace=self.mod.TraceInfo(
                files={"vllm/model_executor/models/kimi_audio.py"}
            ),
            source_tags={"git-trace", "existing-doc"},
        )

        bundles = self.mod.retain_existing_card_fallbacks(
            "vllm", [failed], {36127: card_en}, {36127: card_zh}
        )

        self.assertEqual(len(bundles), 1)
        fallback = bundles[0]
        self.assertIn("existing-card-fallback", fallback.source_tags)
        self.assertEqual(fallback.info["title"], "Add Kimi Audio")
        self.assertEqual(fallback.info["merged_at"], "2026-03-11T00:00:00Z")
        rendered_en = self.mod.render_history_en(
            "vllm",
            "kimi",
            [],
            fallback.trace and {36127: fallback.trace},
            bundles,
            0,
            {36127: card_en},
            {
                36127: (
                    "| 2026-03-11 | [#36127](https://github.com/vllm-project/"
                    "vllm/pull/36127) | merged | Add Kimi Audio | `kimi_audio.py` |"
                )
            },
        )
        rendered_zh = self.mod.render_history_zh(
            "vllm",
            "kimi",
            [],
            {36127: fallback.trace},
            bundles,
            0,
            {36127: card_zh},
            {
                36127: (
                    "| 2026-03-11 | [#36127](https://github.com/vllm-project/"
                    "vllm/pull/36127) | merged | 支持 Kimi Audio | `kimi_audio.py` |"
                )
            },
        )

        self.assertIn("preserved implementation details", rendered_en)
        self.assertIn("Metadata refresh note", rendered_en)
        self.assertIn("保留实现细节", rendered_zh)
        self.assertIn("元数据刷新说明", rendered_zh)
        self.assertIn("| 2026-03-11 | [#36127]", rendered_en)

    def test_extract_existing_cards_and_timeline_rows(self):
        history = """\
## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-03-11 | [#36127](https://github.com/vllm-project/vllm/pull/36127) | merged | Kimi Audio | `kimi_audio.py` |

## Per-PR Diff Audit Cards

### PR #36127 - Kimi Audio

- Link: https://github.com/vllm-project/vllm/pull/36127
- Status/date: merged / 2026-03-11
- Key implementation: keep me.

## Coverage Gap Review
"""
        def fake_run(command, *_args, **_kwargs):
            if command[:3] == ["git", "merge-base", "HEAD"]:
                return "base-sha\n"
            return history

        with mock.patch.object(self.mod, "run", side_effect=fake_run):
            cards = self.mod.extract_existing_cards("vllm", "kimi", "en")
            rows = self.mod.extract_existing_timeline_rows("vllm", "kimi", "en")

        self.assertEqual(set(cards), {36127})
        self.assertIn("Key implementation: keep me", cards[36127])
        self.assertEqual(set(rows), {36127})
        self.assertIn("Kimi Audio", rows[36127])

    def test_top_files_keep_supporting_runtime_when_trace_hits_a_test(self):
        bundle = self.mod.PRBundle(
            framework="vllm",
            repo="vllm-project/vllm",
            number=49963,
            info={},
            files=[
                {
                    "filename": "tests/entrypoints/test_jina.py",
                    "status": "added",
                    "changes": 59,
                },
                {
                    "filename": "vllm/entrypoints/pooling/scoring/io_processor.py",
                    "status": "modified",
                    "changes": 11,
                },
            ],
            trace=self.mod.TraceInfo(files={"tests/entrypoints/test_jina.py"}),
            source_tags={"git-trace"},
        )

        self.assertEqual(
            [file["filename"] for file in self.mod.top_files(bundle)],
            [
                "tests/entrypoints/test_jina.py",
                "vllm/entrypoints/pooling/scoring/io_processor.py",
            ],
        )

    def test_top_files_stay_focused_when_trace_hits_runtime(self):
        bundle = self.mod.PRBundle(
            framework="vllm",
            repo="vllm-project/vllm",
            number=1,
            info={},
            files=[
                {
                    "filename": "vllm/model_executor/models/model.py",
                    "status": "modified",
                    "changes": 10,
                },
                {
                    "filename": "vllm/shared/helper.py",
                    "status": "modified",
                    "changes": 100,
                },
            ],
            trace=self.mod.TraceInfo(
                files={"vllm/model_executor/models/model.py"}
            ),
            source_tags={"git-trace"},
        )

        self.assertEqual(
            [file["filename"] for file in self.mod.top_files(bundle)],
            ["vllm/model_executor/models/model.py"],
        )


if __name__ == "__main__":
    unittest.main()
