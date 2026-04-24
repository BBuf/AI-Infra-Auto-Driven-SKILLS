from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
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

    def test_mapping_formal_overlap_uses_matching_formal_stage_payload(self) -> None:
        args = SimpleNamespace(
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
            num_steps=5,
            profile_by_stage=True,
            merge_profiles=False,
            probe_requests=0,
            probe_prompt="x",
            probe_max_new_tokens=None,
            probe_delay=0.0,
            start_step=None,
            framework="sglang",
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


if __name__ == "__main__":
    unittest.main()
