from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "llm-torch-profiler-analysis"
    / "scripts"
)
SCRIPT = SCRIPT_DIR / "analyze_llm_torch_profile.py"


def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "analyze_llm_torch_profile", SCRIPT
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT_DIR))


class KernelStub:
    def __init__(self, name: str, stage: str) -> None:
        self.name = name
        self.canonical_name = name
        self.category = "compute"
        self.stage = stage
        self.dur = 100.0
        self.ts = 0.0


class LlmTorchProfilerAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def make_args(self, **overrides) -> SimpleNamespace:
        values = {
            "framework": "sglang",
            "num_steps": 5,
            "profile_by_stage": True,
            "merge_profiles": False,
            "probe_requests": 0,
            "probe_prompt": "x",
            "probe_max_new_tokens": None,
            "probe_delay": 0.0,
            "warmup_steps": self.mod.DEFAULT_WARMUP_STEPS,
            "profile_workload": "legacy",
            "prefill_input_len": self.mod.DEFAULT_PREFILL_INPUT_LEN,
            "prefill_output_len": self.mod.DEFAULT_PREFILL_OUTPUT_LEN,
            "decode_input_len": self.mod.DEFAULT_DECODE_INPUT_LEN,
            "decode_output_len": self.mod.DEFAULT_DECODE_OUTPUT_LEN,
            "start_step": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_live_capture_requires_new_artifact_and_accepts_overwrite(self):
        common = sys.modules["profile_common"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve()
            trace = path / "fixed.trace.json"
            trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "old",
                                "ts": 0,
                                "dur": 1,
                            }
                        ]
                    }
                )
            )
            before = common.snapshot_trace_files(path)
            with self.assertRaises(TimeoutError):
                common.wait_for_profiler_artifact(path, timeout_s=0, before=before)
            partial = path / "in-progress.pt.trace.json"
            partial.write_text('{"traceEvents": [')
            with self.assertRaises(TimeoutError):
                common.wait_for_profiler_artifact(path, timeout_s=0, before=before)
            trace.write_text(trace.read_text().replace("old", "new"))
            os.utime(trace, ns=(before[trace][0] + 1000000, before[trace][0] + 1000000))
            self.assertEqual(common.changed_trace_files(path, before), [trace])
            self.assertEqual(
                common.wait_for_profiler_artifact(path, timeout_s=0, before=before),
                path,
            )

    def test_openai_endpoint_alone_does_not_identify_vllm(self):
        common = sys.modules["profile_common"]
        for owner, expected in [
            (None, None),
            ("tensorrt_llm", "trtllm"),
            ("vllm", "vllm"),
        ]:
            with (
                self.subTest(owner=owner),
                mock.patch.object(
                    common,
                    "try_get_json",
                    side_effect=[None, {"data": [{"id": "model", "owned_by": owner}]}],
                ),
            ):
                self.assertEqual(
                    common.detect_framework_from_url("http://server"), expected
                )

    def test_trtllm_rank_selection_prefers_rank_zero(self):
        common = sys.modules["profile_common"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve()
            for rank in [0, 1]:
                trace = path / f"trtllm-trace-run-rank-{rank}.json"
                trace.write_text(json.dumps({"traceEvents": []}))
            selected, _ = common.discover_trace_targets(path, all_traces=False)
            self.assertEqual(
                [p.name for p in selected], ["trtllm-trace-run-rank-0.json"]
            )

    def test_capture_isolates_overwritten_trace_without_moving_server_file(self):
        common = sys.modules["profile_common"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixed = root / "fixed.trace.json"
            old = root / "old.trace.json"
            fixed.write_text(json.dumps({"traceEvents": []}))
            old.write_text(fixed.read_text())
            payloads = []

            def post(url, payload=None, **kwargs):
                if url.endswith("/start_profile"):
                    payloads.append(payload)
                if url.endswith("/stop_profile"):
                    fixed.write_text(json.dumps({"traceEvents": [{"name": "fresh"}]}))

            with (
                mock.patch.object(common, "post_json", side_effect=post),
                mock.patch.object(
                    common, "discover_openai_model", return_value="model"
                ),
            ):
                captured = common.run_remote_profiler(
                    "http://server",
                    str(root),
                    "trtllm",
                    common.ProbePlan("x", 1, 0, 1, 0),
                    0,
                    stage="decode",
                )
            self.assertEqual(payloads[0]["output_dir"], str(root))
            self.assertEqual(common.parse_stage(captured / fixed.name), "decode")
            self.assertEqual((captured / fixed.name).read_bytes(), fixed.read_bytes())
            self.assertFalse((captured / old.name).exists())
            self.assertTrue(old.exists())

    def test_trtllm_live_payload_uses_native_directory_contract(self):
        common = sys.modules["profile_common"]
        payload = common.build_remote_profiler_start_payload(
            "trtllm", Path("/shared/traces"), None, "decode"
        )
        self.assertEqual(
            payload, {"output_dir": "/shared/traces", "activities": ["CPU", "GPU"]}
        )
        self.assertIsNone(
            common.build_remote_profiler_start_payload(
                "vllm", Path("/shared/traces"), None, "decode"
            )
        )

    def test_dsv41_kernel_categories_keep_math_and_communication_distinct(self):
        classify = self.mod.kernel_helpers.classify_kernel
        for name, expected in [
            ("_router_triton_kernel", "router"),
            ("_hc_mix_reduce_sinkhorn_kernel", "hyperconnection"),
            ("void sglang::wo_a_mega::wo_a_kernel", "attention_projection"),
            ("void sglang::moe_finalize_all_reduce_kernel", "communication"),
            ("kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel", "norm"),
            (
                "kernel_cutlass_kernel_flashinferquantizationMxfp8QuantizeKernel",
                "quantize",
            ),
            (
                "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel",
                "gemm",
            ),
        ]:
            with self.subTest(name=name):
                self.assertEqual(classify(name), expected)

    def test_auto_framework_does_not_guess_sglang(self) -> None:
        profile_common = sys.modules["profile_common"]
        with self.assertRaisesRegex(ValueError, "--framework"):
            profile_common.resolve_framework(
                "auto",
                input_path=Path("/tmp/framework-neutral-trace"),
            )

    def test_mapping_formal_overlap_uses_matching_formal_stage_payload(self) -> None:
        args = self.make_args(
            input=None,
            url=None,
            mapping_input="mapping",
            mapping_url=None,
            formal_input="formal",
            formal_url=None,
            output_dir=None,
            mapping_output_dir=None,
            formal_output_dir=None,
            profile_prefix=None,
            mapping_profile_prefix="mapping-trace",
            formal_profile_prefix="formal-trace",
            pid_substring=None,
            kernel_table_limit=0,
            overlap_table_limit=0,
        )

        mapping_trace = Path("mapping.trace.json")
        formal_extend_trace = Path("formal-extend.trace.json")
        formal_decode_trace = Path("formal-decode.trace.json")
        payloads_seen_by_overlap = []

        def fake_resolve_profile_targets(*, label, **_kwargs):
            if label == "mapping":
                return [mapping_trace], None, "sglang"
            return [formal_extend_trace, formal_decode_trace], None, "sglang"

        def fake_extract_trace_data(_trace):
            call_index = fake_extract_trace_data.call_count
            fake_extract_trace_data.call_count += 1
            if call_index == 0:
                kernel = KernelStub("k_mapping", "all")
            elif call_index == 1:
                kernel = KernelStub("k_extend", "extend")
            else:
                kernel = KernelStub("k_decode", "decode")
            return [kernel], [], {}, [], None, 100.0

        fake_extract_trace_data.call_count = 0

        def fake_group_kernels_by_stage(kernels, _default_stage):
            return {kernels[0].stage: kernels}

        def fake_build_stage_payload(_site_stats, kernel_categories):
            kernel_name = next(iter(kernel_categories))
            return {"kernels": {kernel_name: {"best_location": kernel_name}}}

        def fake_build_overlap_stage_bundle_map(_traces, **_kwargs):
            return {
                "extend": SimpleNamespace(events=[], raw_events=[], server_args=None),
                "decode": SimpleNamespace(events=[], raw_events=[], server_args=None),
            }

        def fake_merge_source_map_from_kernel_payload(source_map, stage_payload):
            payloads_seen_by_overlap.append(stage_payload)
            return source_map

        with (
            mock.patch.object(
                self.mod, "resolve_profile_targets", fake_resolve_profile_targets
            ),
            mock.patch.object(self.mod, "load_trace_json", return_value={}),
            mock.patch.object(
                self.mod,
                "build_overlap_stage_bundle_map",
                fake_build_overlap_stage_bundle_map,
            ),
            mock.patch.object(
                self.mod.kernel_helpers, "extract_trace_data", fake_extract_trace_data
            ),
            mock.patch.object(
                self.mod.kernel_helpers,
                "group_kernels_by_stage",
                fake_group_kernels_by_stage,
            ),
            mock.patch.object(
                self.mod.kernel_helpers, "build_cpu_op_index", return_value={}
            ),
            mock.patch.object(
                self.mod.kernel_helpers, "build_launch_index", return_value={}
            ),
            mock.patch.object(
                self.mod.kernel_helpers, "aggregate_kernel_sites", return_value={}
            ),
            mock.patch.object(
                self.mod.kernel_helpers, "build_stage_payload", fake_build_stage_payload
            ),
            mock.patch.object(self.mod.kernel_helpers, "aggregate", return_value={}),
            mock.patch.object(
                self.mod.kernel_helpers, "build_kernel_rows", return_value=[]
            ),
            mock.patch.object(
                self.mod.kernel_helpers, "detect_fusion_opportunities", return_value=[]
            ),
            mock.patch.object(
                self.mod.overlap_helpers,
                "analyze_overlap",
                return_value={"total_busy_us": 100.0},
            ),
            mock.patch.object(
                self.mod.overlap_helpers, "aggregate_events", return_value={}
            ),
            mock.patch.object(
                self.mod.overlap_helpers, "build_kernel_source_map", return_value={}
            ),
            mock.patch.object(
                self.mod.overlap_helpers,
                "merge_source_map_from_kernel_payload",
                fake_merge_source_map_from_kernel_payload,
            ),
            mock.patch.object(
                self.mod.overlap_helpers, "build_action_rows", return_value=[]
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(self.mod.run_triage(args), 0)

        self.assertEqual(
            [set(payload["kernels"]) for payload in payloads_seen_by_overlap],
            [{"k_extend"}, {"k_decode"}],
        )

    def test_stage_parser_uses_parent_workload_directories(self) -> None:
        self.assertEqual(
            self.mod.parse_stage(Path("/tmp/live_profile/prefill/rank0.trace.json")),
            "extend",
        )
        self.assertEqual(
            self.mod.parse_stage(Path("/tmp/live_profile/decode/rank0.trace.json")),
            "decode",
        )

    def test_live_capture_forwards_stage_separated_workload_args(self) -> None:
        args = self.make_args(
            num_steps=7,
            probe_prompt="legacy prompt",
            probe_requests=1,
            profile_workload="both",
            prefill_input_len=1234,
            decode_output_len=321,
        )
        target_dir = Path("/tmp/stage-split-profile")
        captured = {}

        def fake_run_profiler(**kwargs):
            captured.update(kwargs)
            return target_dir

        with (
            mock.patch.object(self.mod, "run_profiler", fake_run_profiler),
            mock.patch.object(
                self.mod,
                "discover_trace_targets",
                return_value=([target_dir / "prefill" / "rank0.trace.json"], None),
            ),
        ):
            traces, _server_args, framework = self.mod.resolve_profile_targets(
                label="input",
                input_path=None,
                url="http://127.0.0.1:30000",
                output_dir=str(target_dir),
                profile_prefix="triage",
                args=args,
            )

        self.assertEqual(framework, "sglang")
        self.assertEqual(traces, [target_dir / "prefill" / "rank0.trace.json"])
        self.assertEqual(captured["profile_workload"], "both")
        self.assertEqual(captured["prefill_input_len"], 1234)
        self.assertEqual(captured["decode_output_len"], 321)
        self.assertEqual(captured["warmup_steps"], 10)

    def test_sglang_stage_workload_warms_up_and_adds_guard_step(self) -> None:
        profile_common = sys.modules["profile_common"]
        calls = []

        def fake_run_sglang_profiler(**kwargs):
            calls.append(kwargs)
            return Path(kwargs["output_dir"])

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(
                profile_common, "run_sglang_profiler", fake_run_sglang_profiler
            ),
        ):
            result = profile_common.run_profiler(
                url="http://127.0.0.1:30000",
                output_dir=tmpdir,
                num_steps=7,
                profile_by_stage=True,
                merge_profiles=False,
                profile_prefix="triage",
                probe_requests=1,
                probe_prompt="legacy prompt",
                probe_max_new_tokens=None,
                probe_delay=0.0,
                warmup_steps=10,
                framework="sglang",
                profile_workload="both",
                prefill_input_len=1234,
                prefill_output_len=1,
                decode_input_len=1,
                decode_output_len=321,
            )
            self.assertEqual(result, Path(tmpdir).resolve())

        self.assertEqual([call["num_steps"] for call in calls], [8, 8])
        self.assertEqual([call["profile_by_stage"] for call in calls], [False, False])
        prefill_plan, decode_plan = [call["probe_plan"] for call in calls]
        self.assertEqual(prefill_plan.capture_requests, 7)
        self.assertEqual(prefill_plan.warmup_requests, 10)
        self.assertEqual(prefill_plan.capture_max_new_tokens, 1)
        self.assertEqual(decode_plan.capture_requests, 1)
        self.assertEqual(decode_plan.warmup_requests, 1)
        self.assertEqual(decode_plan.warmup_max_new_tokens, 10)
        self.assertEqual(decode_plan.capture_max_new_tokens, 321)

    def test_probe_requests_vary_prompt_prefix_to_avoid_prefix_cache(self) -> None:
        profile_common = sys.modules["profile_common"]
        calls = []

        def fake_send_probe_request(**kwargs):
            calls.append(kwargs)

        with mock.patch.object(
            profile_common, "send_probe_request", fake_send_probe_request
        ):
            profile_common.send_probe_requests(
                url="http://127.0.0.1:30000",
                prompt="profile profile profile",
                max_new_tokens=1,
                request_count=3,
                framework="sglang",
                sampling_seed_offset=10,
            )

        prompts = [call["prompt"] for call in calls]
        self.assertEqual(len(prompts), 3)
        self.assertEqual(len(set(prompts)), 3)
        self.assertEqual([call["sampling_seed"] for call in calls], [10, 11, 12])
        self.assertEqual(
            prompts,
            [
                "profile_probe_10 profile profile",
                "profile_probe_11 profile profile",
                "profile_probe_12 profile profile",
            ],
        )

    def test_omni_source_roots_normalize_to_their_own_prefix(self) -> None:
        profile_common = sys.modules["profile_common"]
        for path, expected in (
            (
                "/work/sglang-omni/sglang_omni/models/qwen3_omni.py",
                "sglang_omni/models/qwen3_omni.py",
            ),
            (
                "/work/sglang-omni/sglang_omni_router/dispatch.py",
                "sglang_omni_router/dispatch.py",
            ),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    profile_common.normalize_repo_relative_path(path), expected
                )

    def test_omni_source_roots_win_the_kernel_source_ladder(self) -> None:
        kernel = self.mod.kernel_helpers
        omni_frame = "/work/sglang-omni/sglang_omni/models/qwen3_omni.py(214): forward"
        torch_frame = (
            "/usr/lib/python3.11/site-packages/torch/nn/modules/"
            "linear.py(125): forward"
        )
        omni = kernel.normalize_source_location(omni_frame)
        self.assertTrue(kernel.is_preferred_source_location(omni))
        self.assertGreater(
            kernel.source_location_priority(omni),
            kernel.source_location_priority(
                kernel.normalize_source_location(torch_frame)
            ),
        )
        self.assertGreater(
            kernel.frame_priority(omni_frame), kernel.frame_priority(torch_frame)
        )

    def test_omni_source_roots_win_the_overlap_scope_ladder(self) -> None:
        overlap = self.mod.overlap_helpers
        omni = overlap.canonicalize_python_scope_name(
            "/work/sglang-omni/sglang_omni/models/qwen3_omni.py(214): forward"
        )
        torch_scope = overlap.canonicalize_python_scope_name(
            "/usr/lib/python3.11/site-packages/torch/nn/modules/"
            "linear.py(125): forward"
        )
        self.assertGreater(
            overlap.source_scope_priority(omni),
            overlap.source_scope_priority(torch_scope),
        )
        self.assertEqual(overlap.choose_best_scope((omni, torch_scope)), omni)

    def test_overlap_scope_prefers_model_code_over_the_torch_runtime(self) -> None:
        overlap = self.mod.overlap_helpers
        torch_scope = overlap.canonicalize_python_scope_name(
            "/usr/lib/python3.11/site-packages/torch/nn/modules/"
            "linear.py(125): forward"
        )
        for frame in (
            "/usr/lib/python3.11/site-packages/transformers/models/whisper/"
            "modeling_whisper.py(310): forward",
            "/usr/lib/python3.11/site-packages/torchaudio/functional/"
            "filtering.py(90): lfilter",
        ):
            with self.subTest(frame=frame):
                owner = overlap.canonicalize_python_scope_name(frame)
                self.assertGreater(
                    overlap.source_scope_priority(owner),
                    overlap.source_scope_priority(torch_scope),
                )
                self.assertEqual(overlap.choose_best_scope((owner, torch_scope)), owner)

    def test_overlap_scope_keeps_an_all_torch_chain_usable(self) -> None:
        overlap = self.mod.overlap_helpers
        chain = tuple(
            overlap.canonicalize_python_scope_name(frame)
            for frame in (
                "/usr/lib/python3.11/site-packages/torch/nn/modules/"
                "linear.py(125): forward",
                "/usr/lib/python3.11/site-packages/torch/functional.py(650): einsum",
            )
        )
        best = overlap.choose_best_scope(chain)
        self.assertIn(best, chain)
        self.assertGreater(overlap.source_scope_priority(best), 0)


if __name__ == "__main__":
    unittest.main()
