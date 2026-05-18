"""Tests for model_profiles module and profile-aware analysis functions."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "llm-pipeline-analysis"
    / "scripts"
)


def _load_module(name, script_path):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, script_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # register so dataclass annotations resolve
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT_DIR))


def load_profiles():
    return _load_module("model_profiles", SCRIPT_DIR / "model_profiles.py")


def load_timeline():
    return _load_module("layer_timeline_analyzer", SCRIPT_DIR / "layer_timeline_analyzer.py")


def load_breakdown():
    return _load_module("layer_kernel_breakdown", SCRIPT_DIR / "layer_kernel_breakdown.py")


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
            kernels.append({"name": f"kernel_{i}", "dur": block_dur, "ts": i * block_dur})
        return kernels

    def test_returns_default_when_too_few_blocks(self):
        gpu = [{"name": "k", "dur": 100, "ts": 0}] * 10
        indices = list(range(3))
        result = self.mod.detect_num_layers(indices, gpu, blocks_per_layer=2,
                                             default_num_layers=43)
        self.assertEqual(result, 43)

    def test_default_num_layers_is_configurable(self):
        gpu = [{"name": "k", "dur": 100, "ts": 0}] * 10
        indices = list(range(3))
        result = self.mod.detect_num_layers(indices, gpu, blocks_per_layer=1,
                                             default_num_layers=7)
        self.assertEqual(result, 7)


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
        gpu = [{"name": f"k{i}", "dur": 10, "ts": i * 10, "args": {}}
               for i in range(20)]
        anchor_indices = [0, 5, 10, 15, 20]
        # This should not crash
        kernels = self.mod.get_layer_kernels(gpu, anchor_indices, 0, 0, 2, profile)
        self.assertTrue(len(kernels) > 0)
        # Should have both "attn" and "ffn" halves
        halves = set(k["half"] for k in kernels)
        self.assertIn("attn", halves)
        self.assertIn("ffn", halves)

    def test_generic_one_block_per_layer(self):
        profile = self.profiles.get_profile("generic")
        gpu = [{"name": f"k{i}", "dur": 10, "ts": i * 10, "args": {}}
               for i in range(20)]
        anchor_indices = [0, 10, 20]
        kernels = self.mod.get_layer_kernels(gpu, anchor_indices, 0, 0, 2, profile)
        # Should have only "full" halves
        halves = set(k["half"] for k in kernels)
        self.assertEqual(halves, {"full"})


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
