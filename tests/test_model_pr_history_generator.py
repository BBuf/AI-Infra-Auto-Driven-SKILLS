"""Consistency tests for the model PR-history generator."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
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
            {"hunyuan3-preview", "moss-vl", "qwen36", "qwen38"}
            <= set(self.mod.FRAMEWORK_MODEL_ORDER["sglang"])
        )
        self.assertTrue(
            {"hunyuan3-preview", "qwen36"}
            <= set(self.mod.FRAMEWORK_MODEL_ORDER["vllm"])
        )
        self.assertNotIn("moss-vl", self.mod.FRAMEWORK_MODEL_ORDER["vllm"])
        self.assertNotIn("qwen38", self.mod.FRAMEWORK_MODEL_ORDER["vllm"])

    def test_generated_inventory_does_not_claim_manual_diff_review(self):
        bundle = self.mod.PRBundle(
            framework="tokenspeed", repo="lightseekorg/tokenspeed", number=1,
            info={"number": 1, "title": "Fuse query preparation", "state": "closed",
                  "merged_at": "2026-09-21T00:00:00Z"},
            files=[{"filename": "kernel.py", "additions": 1, "deletions": 0,
                    "changes": 1, "status": "modified", "patch": "@@ -1 +1 @@\n+x = y"}],
            trace=self.mod.TraceInfo(), source_tags=set(),
        )
        en = self.mod.card_en(bundle, "DeepSeek V4.1")
        zh = self.mod.card_zh(bundle, "DeepSeek V4.1")
        self.assertIn("not a manual audit", en)
        self.assertIn("Manual review pending", en)
        self.assertNotIn("- Diff scope read:", en)
        self.assertNotIn("- Reviewed files:", en)
        self.assertIn("不是人工审计", zh)
        self.assertNotIn("- 已读文件:", zh)

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

        with mock.patch.object(self.mod, "run", side_effect=fake_run) as run, \
             mock.patch.object(Path, "exists", return_value=False):
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

        with mock.patch.object(self.mod, "run", side_effect=fake_run), \
             mock.patch.object(Path, "exists", return_value=False):
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

    def test_new_framework_filters_split_neighboring_generations(self):
        trt_files = [
            "tensorrt_llm/_torch/models/modeling_deepseekv3.py",
            "tensorrt_llm/_torch/models/modeling_deepseekv4.py",
            "cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu",
            "tensorrt_llm/_torch/models/modeling_qwen4_exp.py",
        ]
        self.assertEqual(
            self.mod.selected_files("tensorrt_llm", "deepseek-v4", trt_files),
            ["tensorrt_llm/_torch/models/modeling_deepseekv4.py"],
        )
        self.assertEqual(
            self.mod.selected_files("tensorrt_llm", "kimi", trt_files),
            ["cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu"],
        )
        sglang_files = [
            "python/sglang/srt/models/hunyuan_v4.py",
            "python/sglang/multimodal_gen/runtime/models/dits/hunyuan3d.py",
            "python/sglang/srt/models/qwen4_exp.py",
            "python/sglang/srt/models/bailing_moe_v3.py",
        ]
        self.assertEqual(
            self.mod.selected_files("sglang", "hunyuan4", sglang_files),
            ["python/sglang/srt/models/hunyuan_v4.py"],
        )
        self.assertEqual(
            self.mod.selected_files("sglang", "ling3", sglang_files),
            ["python/sglang/srt/models/bailing_moe_v3.py"],
        )

    def test_deepseek_v41_keeps_only_v41_subjects(self):
        traces = {
            1: self.mod.TraceInfo(subjects={"dsv4.1: vision tower (#1)"}),
            2: self.mod.TraceInfo(subjects={"[DSV4] fix mega moe (#2)"}),
            3: self.mod.TraceInfo(subjects={"DeepSeek-V4.1 cookbook (#3)"}),
        }
        self.assertEqual(
            set(self.mod.filter_traces_by_subject("sglang", "deepseek-v41", traces)),
            {1, 3},
        )

    def test_preamble_keeps_manual_notes_but_drops_embedded_cards(self):
        doc = """# TensorRT-LLM Kimi Model PR Optimization History

## 2026-08-23 Source Head Refresh

Result: PR #16805 is promoted.

### PR #16805 - Fix draft-token accounting

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16805
- Status/date: merged / 2026-07-27

## 2026-06-27 PR Backfill Audit

Filter used in this pass.

## Implementation File Coverage

| File | Git-traced PRs |
"""
        with mock.patch.object(self.mod, "run", return_value=doc), \
             mock.patch.object(Path, "exists", return_value=False):
            preamble = self.mod.extract_preamble("tensorrt_llm", "kimi", "en")
        self.assertIn("Result: PR #16805 is promoted.", preamble)
        self.assertIn("## 2026-06-27 PR Backfill Audit", preamble)
        self.assertNotIn("### PR #16805", preamble)
        self.assertNotIn("Implementation File Coverage", preamble)

    def test_uncommitted_manual_notes_and_cards_survive_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "model-pr-optimization-history"
            path = history / "tokenspeed" / "deepseek-v41" / "README.en.md"
            path.parent.mkdir(parents=True)
            path.write_text("# Model history\n\n## Manual note\n\nReviewed now.\n\n"
                            "## Implementation File Coverage\n\n"
                            "### PR #1 - New review\n\n- Key implementation: manual current.\n")
            committed = ("# Model history\n\n## Implementation File Coverage\n\n"
                         "### PR #1 - Old review\n\n- Key implementation: old.\n\n"
                         "### PR #2 - Historical review\n\n- Key implementation: recovered.\n")
            def fake_run(cmd, *args, **kwargs):
                return "" if cmd[1] == "merge-base" else committed
            with mock.patch.object(self.mod, "ROOT", root), \
                 mock.patch.object(self.mod, "HISTORY_ROOT", history), \
                 mock.patch.object(self.mod, "run", side_effect=fake_run):
                preamble = self.mod.extract_preamble("tokenspeed", "deepseek-v41", "en")
                cards = self.mod.extract_existing_cards("tokenspeed", "deepseek-v41", "en")
            self.assertIn("Reviewed now.", preamble)
            self.assertIn("manual current.", cards[1])
            self.assertNotIn("Old review", cards[1])
            self.assertIn("recovered.", cards[2])

    def test_merged_cards_with_rows_are_reused_without_refetch(self):
        merged = "### PR #1 - A\n\n- Status/date: merged / 2026-01-01\n"
        merged_zh = "### PR #1 - A\n\n- 状态/时间: merged / 2026-01-01\n"
        open_card = "### PR #2 - B\n\n- Status/date: open / 2026-01-02\n"
        row = "| 2026-01-01 | [#1](https://github.com/x/y/pull/1) | merged | A | `a.py` |"
        reuse = self.mod.reusable_existing_numbers(
            {1, 2, 3},
            {1: merged, 2: open_card},
            {1: merged_zh, 2: open_card},
            {1: row, 2: row},
            {1: row},
        )
        self.assertEqual(reuse, {1})

        bundles = self.mod.reused_bundles(
            "sglang", {1}, {}, {1: {"git-trace"}}, {1: merged}, {1: merged_zh}
        )
        rendered = self.mod.render_history_en(
            "sglang", "kimi", [], {}, bundles, 0, {1: merged}, {1: row},
            "Manual addendum.",
        )
        self.assertIn("- Status/date: merged / 2026-01-01", rendered)
        self.assertIn(row, rendered)
        self.assertIn("Manual addendum.\n\n## Implementation File Coverage", rendered)
        self.assertTrue(rendered.startswith("# SGLang Kimi"))


if __name__ == "__main__":
    unittest.main()
