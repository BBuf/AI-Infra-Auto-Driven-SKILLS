"""Tests for model_profiles module and profile-aware analysis functions."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "skills" / "llm-pipeline-analysis" / "scripts"
)
SIM_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "model-compute-simulation"
    / "scripts"
)
CONFIG_INDEX = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "model-compute-simulation"
    / "references"
    / "model-config-index.json"
)
GPU_SPECS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "model-compute-simulation"
    / "references"
    / "gpu-specs.json"
)


def _load_module(name, script_path):
    sys.path.insert(0, str(script_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, script_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # register so dataclass annotations resolve
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_path.parent))


def load_profiles():
    return _load_module("model_profiles", SCRIPT_DIR / "model_profiles.py")


def load_timeline():
    return _load_module(
        "layer_timeline_analyzer", SCRIPT_DIR / "layer_timeline_analyzer.py"
    )


def load_breakdown():
    return _load_module(
        "layer_kernel_breakdown", SCRIPT_DIR / "layer_kernel_breakdown.py"
    )


def load_simulator():
    return _load_module(
        "model_compute_simulator", SIM_SCRIPT_DIR / "model_compute_simulator.py"
    )


def load_compute_extractor():
    return _load_module(
        "extract_compute_flow_from_trace",
        SIM_SCRIPT_DIR / "extract_compute_flow_from_trace.py",
    )


# ---------------------------------------------------------------------------
# Test: ModelProfile data structure and built-in profiles
# ---------------------------------------------------------------------------


class TestModelProfile(unittest.TestCase):
    def setUp(self):
        self.mod = load_profiles()

    def test_builtin_profiles_exist(self):
        for name in ["dsv4_csa_hca", "dsv3_mla", "generic"]:
            p = self.mod.get_profile(name)
            self.assertEqual(p.name, name)

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            self.mod.get_profile("nonexistent")

    def test_dsv4_profile_attributes(self):
        p = self.mod.get_profile("dsv4_csa_hca")
        self.assertEqual(p.anchor_kernel, "mhc_post_tilelang")
        self.assertEqual(p.blocks_per_layer, 2)
        self.assertEqual(p.half_labels, ["attn", "ffn"])
        self.assertEqual(p.default_num_layers, 43)
        self.assertTrue(len(p.category_rules) > 10)

    def test_dsv3_profile_attributes(self):
        p = self.mod.get_profile("dsv3_mla")
        self.assertEqual(p.anchor_kernel, "flash_fwd_mla_combine")
        self.assertEqual(p.blocks_per_layer, 1)
        self.assertEqual(p.half_labels, ["full"])
        self.assertEqual(p.default_num_layers, 61)

    def test_generic_profile_attributes(self):
        p = self.mod.get_profile("generic")
        self.assertIsNone(p.anchor_kernel)
        self.assertEqual(p.blocks_per_layer, 1)
        self.assertEqual(p.half_labels, ["full"])
        self.assertEqual(p.default_num_layers, 1)

    def test_category_rules_have_display_and_key(self):
        for name in ["dsv4_csa_hca", "dsv3_mla", "generic"]:
            p = self.mod.get_profile(name)
            for label, key, rule in p.category_rules:
                self.assertIsInstance(label, str)
                self.assertIsInstance(key, str)
                self.assertTrue(callable(rule))


# ---------------------------------------------------------------------------
# Test: Profile inference
# ---------------------------------------------------------------------------


class TestInferProfile(unittest.TestCase):
    def setUp(self):
        self.mod = load_profiles()

    def test_compress_ratios_infers_dsv4(self):
        config = {"compress_ratios": [0, 0, 4, 128]}
        p = self.mod.infer_profile(config)
        self.assertEqual(p.name, "dsv4_csa_hca")

    def test_kv_lora_rank_infers_dsv3(self):
        config = {"kv_lora_rank": 512}
        p = self.mod.infer_profile(config)
        self.assertEqual(p.name, "dsv3_mla")

    def test_empty_config_infers_generic(self):
        config = {}
        p = self.mod.infer_profile(config)
        self.assertEqual(p.name, "generic")

    def test_normalize_compress_ratios_matches_hidden_layers(self):
        config = {"num_hidden_layers": 4, "compress_ratios": [0, 0, 4, 128]}
        self.assertEqual(self.mod.normalize_compress_ratios(config), [0, 0, 4, 128])

    def test_normalize_compress_ratios_allows_nextn_trailing_ratio(self):
        config = {
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "compress_ratios": [0, 0, 4, 128, 0],
        }
        self.assertEqual(self.mod.normalize_compress_ratios(config), [0, 0, 4, 128])

    def test_normalize_compress_ratios_rejects_unexplained_mismatch(self):
        config = {"num_hidden_layers": 4, "compress_ratios": [0, 0, 4]}
        with self.assertRaises(ValueError):
            self.mod.normalize_compress_ratios(config)

    def test_compress_ratios_takes_priority_over_kv_lora_rank(self):
        config = {"compress_ratios": [0, 4], "kv_lora_rank": 512}
        p = self.mod.infer_profile(config)
        self.assertEqual(p.name, "dsv4_csa_hca")

    def test_empty_compress_ratios_falls_through(self):
        config = {"compress_ratios": [], "kv_lora_rank": 512}
        p = self.mod.infer_profile(config)
        self.assertEqual(p.name, "dsv3_mla")

    def test_zero_kv_lora_rank_falls_through(self):
        config = {"kv_lora_rank": 0}
        p = self.mod.infer_profile(config)
        self.assertEqual(p.name, "generic")


# ---------------------------------------------------------------------------
# Test: Kernel classification with profiles
# ---------------------------------------------------------------------------


class TestClassifyKernel(unittest.TestCase):
    def setUp(self):
        self.mod = load_profiles()
        self.dsv4 = self.mod.get_profile("dsv4_csa_hca")
        self.dsv3 = self.mod.get_profile("dsv3_mla")
        self.generic = self.mod.get_profile("generic")

    def test_dsv4_classifies_mla(self):
        label, key = self.dsv4.category_rules[0][0], self.dsv4.category_rules[0][1]
        for label_i, key_i, rule in self.dsv4.category_rules:
            if rule("flash_fwd_splitkv_mla_kernel"):
                self.assertEqual(key_i, "mla")
                return
        self.fail("MLA kernel not classified by dsv4 profile")

    def test_dsv4_classifies_mhc(self):
        for label, key, rule in self.dsv4.category_rules:
            if rule("mhc_post_tilelang_kernel"):
                self.assertEqual(key, "mhc_post")
                return
        self.fail("MHC kernel not classified by dsv4 profile")

    def test_generic_does_not_classify_mhc(self):
        for label, key, rule in self.generic.category_rules:
            if rule("mhc_post_tilelang_kernel"):
                self.fail("generic profile should not classify MHC kernels")
        # Should fall to "other"
        self.assertTrue(True)

    def test_generic_classifies_allreduce(self):
        for label, key, rule in self.generic.category_rules:
            if rule("ncclAllReduce_bf16_RING_LL"):
                self.assertEqual(key, "allreduce")
                return
        self.fail("AllReduce kernel not classified by generic profile")

    def test_generic_classifies_rmsnorm(self):
        for label, key, rule in self.generic.category_rules:
            if rule("RMSNormKernel"):
                self.assertEqual(key, "rmsnorm")
                return
        self.fail("RMSNorm kernel not classified by generic profile")

    def test_dsv3_classifies_mla(self):
        for label, key, rule in self.dsv3.category_rules:
            if rule("flash_fwd_splitkv_mla_kernel"):
                self.assertEqual(key, "mla")
                return
        self.fail("MLA kernel not classified by dsv3 profile")


# ---------------------------------------------------------------------------
# Test: detect_num_layers with configurable blocks_per_layer
# ---------------------------------------------------------------------------


class TestDetectNumLayers(unittest.TestCase):
    def setUp(self):
        self.mod = load_timeline()

    def _make_gpu_kernels(self, n_blocks, block_dur=1000):
        """Create fake GPU kernel events with anchor kernels at given indices."""
        kernels = []
        for i in range(n_blocks * 3):  # some filler between anchors
            kernels.append(
                {"name": f"kernel_{i}", "dur": block_dur, "ts": i * block_dur}
            )
        return kernels

    def test_returns_default_when_too_few_blocks(self):
        gpu = [{"name": "k", "dur": 100, "ts": 0}] * 10
        indices = list(range(3))
        result = self.mod.detect_num_layers(
            indices, gpu, blocks_per_layer=2, default_num_layers=43
        )
        self.assertEqual(result, 43)

    def test_default_num_layers_is_configurable(self):
        gpu = [{"name": "k", "dur": 100, "ts": 0}] * 10
        indices = list(range(3))
        result = self.mod.detect_num_layers(
            indices, gpu, blocks_per_layer=1, default_num_layers=7
        )
        self.assertEqual(result, 7)


class TestTimelineSelection(unittest.TestCase):
    def setUp(self):
        self.timeline = load_timeline()
        self.profiles = load_profiles()
        self.breakdown = load_breakdown()

    def test_final_layer_label_wins_over_hash_suffix(self):
        self.assertEqual(
            self.timeline.layer_type_label(3, [0, 0, 128, 128], 4, 2),
            ("FINAL", 128),
        )
        self.assertEqual(
            self.breakdown._layer_type_label(128, 3, 4, 2),
            "FINAL",
        )

    def test_select_steady_state_pass_uses_relative_stability(self):
        self.assertEqual(
            self.timeline.select_steady_state_pass([1000.0, 5100.0, 5050.0, 5075.0]),
            1,
        )
        self.assertIsNone(
            self.timeline.select_steady_state_pass([1000.0, 2000.0, 4000.0, 8000.0])
        )

    def test_select_steady_state_pass_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            self.timeline.select_steady_state_pass([1.0, 1.0], stable_pairs=0)

    def test_generic_anchor_falls_back_to_repeated_rmsnorm(self):
        kernels = [{"name": "rms_norm_kernel"} for _ in range(8)]
        self.assertEqual(
            self.timeline.find_anchor_kernel(
                kernels, self.profiles.get_profile("generic")
            ),
            "rms_norm",
        )

    def test_moe_topk_accepts_num_experts_per_token_alias(self):
        fields = self.breakdown.model_architecture_fields(
            {
                "moe": True,
                "num_experts": 128,
                "num_experts_per_token": 8,
            }
        )
        self.assertEqual(fields["top_k"], 8)


# ---------------------------------------------------------------------------
# Test: get_layer_kernels with configurable blocks_per_layer
# ---------------------------------------------------------------------------


class TestGetLayerKernels(unittest.TestCase):
    def setUp(self):
        self.mod = load_breakdown()
        self.profiles = load_profiles()

    def test_dsv4_two_blocks_per_layer(self):
        profile = self.profiles.get_profile("dsv4_csa_hca")
        # 2 layers, 2 blocks/layer = 4 anchor blocks, plus one more as boundary
        # Create 5 anchor positions (4 blocks + 1 end boundary)
        gpu = [
            {"name": f"k{i}", "dur": 10, "ts": i * 10, "args": {}} for i in range(20)
        ]
        anchor_indices = [0, 5, 10, 15, 20]
        # This should not crash
        kernels = self.mod.get_layer_kernels(gpu, anchor_indices, 0, 0, 2, profile)
        self.assertTrue(len(kernels) > 0)
        self.assertEqual([k["idx"] for k in kernels], list(range(10)))
        # Should have both "attn" and "ffn" halves
        halves = set(k["half"] for k in kernels)
        self.assertIn("attn", halves)
        self.assertIn("ffn", halves)

    def test_generic_one_block_per_layer(self):
        profile = self.profiles.get_profile("generic")
        gpu = [
            {"name": f"k{i}", "dur": 10, "ts": i * 10, "args": {}} for i in range(20)
        ]
        anchor_indices = [0, 10, 20]
        kernels = self.mod.get_layer_kernels(gpu, anchor_indices, 0, 0, 2, profile)
        self.assertEqual([k["idx"] for k in kernels], list(range(10)))
        # Should have only "full" halves
        halves = set(k["half"] for k in kernels)
        self.assertEqual(halves, {"full"})


class TestHotKernelOutput(unittest.TestCase):
    def setUp(self):
        self.mod = load_breakdown()
        self.profiles = load_profiles()
        self.profile = self.profiles.get_profile("generic")

    def _kernels(self):
        return [
            {"idx": 0, "half": "full", "name": "short", "ts": 0, "dur": 10},
            {"idx": 1, "half": "full", "name": "hot_a", "ts": 10, "dur": 50},
            {"idx": 2, "half": "full", "name": "hot_b", "ts": 20, "dur": 50},
            {"idx": 3, "half": "full", "name": "tiny", "ts": 30, "dur": 5},
        ]

    def test_hot_kernel_order_sorts_by_duration_then_trace_order(self):
        ordered = self.mod.hot_kernel_order(self._kernels())
        self.assertEqual(
            [k["name"] for k in ordered], ["hot_a", "hot_b", "short", "tiny"]
        )

    def test_json_kernel_list_is_hotness_sorted_without_changing_wall_time(self):
        payload = json.loads(
            self.mod.format_json_output(
                self._kernels(),
                layer_id=0,
                fwd_pass=0,
                compress_ratios=[],
                num_layers=1,
                profile=self.profile,
            )
        )

        self.assertEqual(
            [k["name"] for k in payload["kernels"]],
            ["hot_a", "hot_b", "short", "tiny"],
        )
        self.assertEqual(payload["metadata"]["wall_us"], 35)

    def test_json_comparison_is_one_machine_readable_document(self):
        comparison = [
            {
                "idx": 4,
                "half": "full",
                "name": "comparison_only",
                "ts": 50,
                "dur": 5,
            }
        ]
        payload = json.loads(
            self.mod.format_json_comparison(
                self._kernels(),
                0,
                comparison,
                1,
                2,
                [0, 4],
                2,
                self.profile,
            )
        )
        self.assertEqual(payload["primary"]["metadata"]["layer_id"], 0)
        self.assertEqual(payload["comparison"]["metadata"]["layer_id"], 1)
        self.assertEqual(
            payload["kernel_diff"]["only_comparison"],
            ["comparison_only"],
        )
        self.assertEqual(payload["kernel_diff"]["common_count"], 0)


# ---------------------------------------------------------------------------
# Test: layer timeline classification and half-open boundaries
# ---------------------------------------------------------------------------


class TestLayerTimelineInfo(unittest.TestCase):
    def setUp(self):
        self.mod = load_timeline()
        self.profiles = load_profiles()
        self.profile = self.profiles.get_profile("dsv4_csa_hca")

    def test_get_layer_info_uses_machine_keys_and_half_open_blocks(self):
        gpu = [
            {"name": f"k{i}", "dur": 10, "ts": i * 10, "args": {}} for i in range(20)
        ]
        gpu[1]["name"] = "flash_fwd_splitkv_mla_kernel"
        gpu[2]["name"] = "mhc_post_tilelang_kernel"
        gpu[9]["name"] = "ncclAllReduce_bf16_RING_LL"
        gpu[10]["name"] = "ncclAllReduce_bf16_RING_LL"
        anchor_indices = [0, 5, 10, 15, 20]

        info = self.mod.get_layer_info(gpu, anchor_indices, 0, 0, 2, self.profile)

        self.assertEqual(info["kernels"], 10)
        self.assertEqual(info["total"], 100)
        self.assertEqual(info["mla"], 10)
        self.assertEqual(info["mhc_post"], 10)
        self.assertEqual(self.mod.prefix_total(info, "mhc"), 10)
        self.assertEqual(info["ar_count"], 1)
        self.assertEqual(info["allreduce"], 10)

    def test_print_layer_detail_smoke(self):
        gpu = [
            {"name": f"k{i}", "dur": 10, "ts": i * 10, "args": {}} for i in range(20)
        ]
        gpu[1]["name"] = "flash_fwd_splitkv_mla_kernel"
        anchor_indices = [0, 5, 10, 15, 20]

        out = io.StringIO()
        with redirect_stdout(out):
            self.mod.print_layer_detail(
                gpu,
                anchor_indices,
                0,
                2,
                [0, 4],
                0,
                0,
                self.profile,
            )

        output = out.getvalue()
        self.assertIn("Forward Pass #0", output)
        self.assertIn("FIRST", output)


# ---------------------------------------------------------------------------
# Test: model config consistency for compress_ratios
# ---------------------------------------------------------------------------


class TestModelConfigIndex(unittest.TestCase):
    def test_new_public_model_configs_have_simulator_dimensions(self):
        configs = json.loads(CONFIG_INDEX.read_text())
        required = {
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "attention_type",
            "moe",
            "num_experts",
            "num_experts_per_tok",
            "routed_expert_intermediate_size",
        }

        for key in ["minimax-m3", "qwen3.6-35b-a3b"]:
            with self.subTest(model=key):
                self.assertIn(key, configs)
                self.assertFalse(required - configs[key].keys())

    def test_minimax_m3_dense_prefix_and_qwen36_hybrid_layers_build(self):
        sim = load_simulator()
        configs = json.loads(CONFIG_INDEX.read_text())

        minimax = configs["minimax-m3"]
        dense_ops = sim.build_layer_ops(minimax, 1, 128, 8, 8, layer_idx=0)
        moe_ops = sim.build_layer_ops(minimax, 1, 128, 8, 8, layer_idx=3)
        self.assertIn("ffn_swiglu", {op.name for op in dense_ops})
        self.assertNotIn("sparse_index_qk_proj", {op.name for op in dense_ops})
        self.assertNotIn("routed_experts_swiglu", {op.name for op in dense_ops})
        self.assertIn("sparse_index_qk_proj", {op.name for op in moe_ops})
        self.assertIn("routed_experts_swiglu", {op.name for op in moe_ops})

        qwen36 = configs["qwen3.6-35b-a3b"]
        linear_ops = sim.build_layer_ops(qwen36, 1, 128, 1, 1, layer_idx=0)
        full_ops = sim.build_layer_ops(qwen36, 1, 128, 1, 1, layer_idx=3)
        self.assertIn("gated_delta_attention", {op.name for op in linear_ops})
        self.assertNotIn("attn_score", {op.name for op in linear_ops})
        self.assertIn("attn_score", {op.name for op in full_ops})
        self.assertEqual(
            next(op for op in linear_ops if op.name == "linear_in_proj_qkvz").shape_out,
            "[1×128×12288]",
        )

    def test_new_model_aliases_resolve_checkpoint_names(self):
        sim = load_simulator()
        configs = json.loads(CONFIG_INDEX.read_text())

        self.assertEqual(
            sim.resolve_model("MiniMaxAI/MiniMax-M3-MXFP8", configs)["display_name"],
            "MiniMax-M3",
        )
        self.assertEqual(
            sim.resolve_model("Qwen/Qwen3.6-35B-A3B-FP8", configs)["display_name"],
            "Qwen3.6-35B-A3B",
        )
        self.assertEqual(
            sim.resolve_model("Qwen/Qwen3.8-27B", configs)["display_name"],
            "Qwen3.8-27B",
        )

    def test_compress_ratios_lengths_are_explained(self):
        configs = json.loads(CONFIG_INDEX.read_text())
        for key, cfg in configs.items():
            ratios = cfg.get("compress_ratios")
            if not ratios:
                continue
            n_layers = cfg["num_hidden_layers"]
            nextn_layers = cfg.get("num_nextn_predict_layers", 0)
            allowed_lengths = {n_layers}
            if nextn_layers:
                allowed_lengths.add(n_layers + nextn_layers)
            self.assertIn(
                len(ratios),
                allowed_lengths,
                f"{key}: compress_ratios length is not explained by hidden/nextn layers",
            )

    def test_simulator_normalizes_nextn_trailing_ratio(self):
        sim = load_simulator()
        cfg = {
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "compress_ratios": [0, 0, 4, 128, 0],
        }
        self.assertEqual(sim.normalize_compress_ratios(cfg), [0, 0, 4, 128])

    def test_gpu_specs_include_local_accelerators(self):
        specs = json.loads(GPU_SPECS.read_text())
        for key in ["h20", "h100-sxm-80gb", "h200-sxm-141gb", "b200-sxm-180gb"]:
            self.assertIn(key, specs)
            self.assertGreater(specs[key]["bf16_tflops"], 0)
            self.assertGreater(specs[key]["fp8_tflops"], 0)
            self.assertGreater(specs[key]["memory_bw_tb_s"], 0)
            self.assertGreater(specs[key]["hbm_gb"], 0)

    def test_gpu_aliases_resolve_local_short_names(self):
        sim = load_simulator()
        specs = json.loads(GPU_SPECS.read_text())
        self.assertEqual(
            sim.resolve_gpu("h100", specs)["display_name"], "NVIDIA H100 SXM 80GB"
        )
        self.assertEqual(
            sim.resolve_gpu("h200", specs)["display_name"], "NVIDIA H200 SXM 141GB"
        )
        self.assertEqual(
            sim.resolve_gpu("b200", specs)["display_name"], "NVIDIA B200 SXM 180GB"
        )


class TestMeasuredComputeFlow(unittest.TestCase):
    def test_kernel_flow_drives_summary_mfu_and_keeps_json_pure(self):
        config = json.loads(CONFIG_INDEX.read_text())["deepseek-v4-flash"]
        kernel_flow = {
            "metadata": {
                "total_dur_us": 2000,
                "compress_ratio": 0,
            },
            "category_summary": {
                "gemm_bf16": {
                    "dur_us": 2000,
                    "count": 1,
                }
            },
            "kernels": [
                {
                    "name": "gemm",
                    "simplified_name": "gemm",
                    "dur_us": 2000,
                    "category": "gemm_bf16",
                }
            ],
        }
        result = subprocess.run(
            [
                sys.executable,
                str(SIM_SCRIPT_DIR / "model_compute_simulator.py"),
                "deepseek-v4-flash",
                "--gpu",
                "b200",
                "--kernel-flow",
                json.dumps(kernel_flow),
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["measured_ms"],
            2.0 * config["num_hidden_layers"],
        )
        self.assertIsNotNone(payload["mfu_pct"])
        self.assertEqual(payload["kernel_flow"]["metadata"]["total_dur_us"], 2000)

    def test_trace_template_comparison_uses_semantic_families(self):
        mod = load_compute_extractor()
        self.assertEqual(
            mod.canonical_trace_op_family("aten::mm", "attention"),
            "matmul",
        )
        self.assertEqual(
            mod.canonical_template_op_family("q_proj", "attention"),
            "matmul",
        )
        self.assertEqual(
            mod.canonical_trace_op_family("aten::rms_norm", "norm"),
            "norm",
        )
        self.assertEqual(
            mod.canonical_template_op_family("rmsnorm", "norm"),
            "norm",
        )

    def test_positive_min_flops_excludes_unknown_zero_flop_ops(self):
        mod = load_compute_extractor()
        events = [
            {
                "cat": "cpu_op",
                "name": "aten::silu",
                "ts": 1,
                "dur": 1,
                "pid": 1,
                "tid": 1,
                "args": {"Input Dims": [[8, 8]]},
            }
        ]
        self.assertEqual(mod.extract_compute_flow(events, min_flops=1), [])


# ---------------------------------------------------------------------------
# Test: simplify_name uses profile rules
# ---------------------------------------------------------------------------


class TestSimplifyName(unittest.TestCase):
    def setUp(self):
        self.mod = load_breakdown()
        self.profiles = load_profiles()

    def test_dsv4_simplifies_mhc(self):
        profile = self.profiles.get_profile("dsv4_csa_hca")
        result = self.mod.simplify_name("mhc_post_tilelang_kernel", profile)
        self.assertEqual(result, "mhc_post_tilelang")

    def test_generic_simplifies_rmsnorm(self):
        profile = self.profiles.get_profile("generic")
        result = self.mod.simplify_name("norm::RMSNormKernel", profile)
        self.assertEqual(result, "RMSNormKernel")

    def test_all_profiles_truncate_long_names(self):
        profile = self.profiles.get_profile("generic")
        long_name = "x" * 100
        result = self.mod.simplify_name(long_name, profile)
        self.assertLessEqual(len(result), 80)


if __name__ == "__main__":
    unittest.main()
