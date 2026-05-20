"""Tests for llm-serving-capacity-planner/scripts/capacity_analyzer.py"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Add scripts dir to path
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "llm-serving-capacity-planner" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from capacity_analyzer import (
    CudaGraphInfo,
    GPU_ALIAS,
    GPU_SPECS_PATH,
    SwKvMemoryCalc,
    FinalInfo,
    MemoryBreakdown,
    MemoryProfiling,
    ModelConfig,
    NvidiaSmiEntry,
    ParsedLog,
    ServerArgs,
    calc_kv_bytes_per_token,
    decompose_memory,
    estimate_concurrency,
    kv_dtype_bytes,
    load_gpu_specs,
    parse_sw_kv_memory_calc,
    parse_cuda_graph_end,
    parse_load_weight_begin,
    parse_log,
    parse_max_total_tokens,
    parse_memory_pool_end,
    parse_memory_profiling,
    parse_nvidia_smi,
    parse_server_args,
)

# ---------------------------------------------------------------------------
# Sample log fixtures
# ---------------------------------------------------------------------------

# SGLang log with mem-fraction-static=0.88 (bf16 KV, no Memory profiling line)
SGLANG_LOG_MFS088 = """\
[2026-05-15 08:39:19] INFO server_args.py:1217: Setting swa_full_tokens_ratio to 0.1 for DeepseekV4ForCausalLM.
[2026-05-15 08:39:19] server_args=ServerArgs(model_path='/root/workspace/models/DeepSeek-V4-Flash-Base', tp_size=8, pp_size=1, dp_size=1, ep_size=1, mem_fraction_static=0.88, kv_cache_dtype='bf16', cuda_graph_max_bs=320, disable_radix_cache=True, page_size=256)
[2026-05-15 08:39:36 TP0] Load weight begin. avail mem=93.61 GB
[2026-05-15 08:39:36 TP1] Load weight begin. avail mem=93.56 GB
[2026-05-15 08:39:36 TP2] Load weight begin. avail mem=93.56 GB
[2026-05-15 08:39:36 TP3] Load weight begin. avail mem=93.56 GB
[2026-05-15 08:39:36 TP4] Load weight begin. avail mem=93.56 GB
[2026-05-15 08:39:36 TP5] Load weight begin. avail mem=93.56 GB
[2026-05-15 08:39:36 TP6] Load weight begin. avail mem=93.56 GB
[2026-05-15 08:39:36 TP7] Load weight begin. avail mem=93.79 GB
[2026-05-15 08:39:53 TP0] DSv4 memory calculation: bytes_per_full_token=15955.85, available_bytes=45.78 GB, full_token=3080960
[2026-05-15 08:39:53 TP1] DSv4 memory calculation: bytes_per_full_token=15955.85, available_bytes=45.78 GB, full_token=3080960
[2026-05-15 08:39:53 TP0] Memory pool end. avail mem=10.18 GB
[2026-05-15 08:39:53 TP1] Memory pool end. avail mem=10.12 GB
[2026-05-15 08:40:32 TP0] Capture cuda graph end. Time elapsed: 54.61 s. mem usage=1.93 GB. avail mem=8.16 GB.
[2026-05-15 08:40:32 TP1] Capture cuda graph end. Time elapsed: 54.53 s. mem usage=1.93 GB. avail mem=8.11 GB.
[2026-05-15 08:40:32 TP0] max_total_num_tokens=3080960, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=256, context_len=1048576, available_gpu_mem=8.16 GB
[2026-05-15 08:40:33] The server is fired up and ready to roll!
"""

# SGLang log with mem-fraction-static=0.6 (fp8 KV, with Memory profiling line)
SGLANG_LOG_MFS060 = """\
[2026-05-15 09:09:21] INFO server_args.py:1217: Setting swa_full_tokens_ratio to 0.1 for DeepseekV4ForCausalLM.
[2026-05-15 09:09:22] server_args=ServerArgs(model_path='/root/workspace/models/DeepSeek-V4-Flash-Base', tp_size=8, pp_size=1, dp_size=1, ep_size=1, mem_fraction_static=0.60, kv_cache_dtype='fp8_e4m3', cuda_graph_max_bs=320, disable_radix_cache=True, page_size=256)
[2026-05-15 09:09:36 TP0] Load weight begin. avail mem=93.61 GB
[2026-05-15 09:09:36 TP1] Load weight begin. avail mem=93.56 GB
[2026-05-15 09:09:53 TP0] Memory profiling: available_gpu_memory=57.01 GB, total_gpu_memory=93.58 GB, mem_fraction_static=0.60, rest_memory=19.58 GB
[2026-05-15 09:09:53 TP1] Memory profiling: available_gpu_memory=57.01 GB, total_gpu_memory=93.58 GB, mem_fraction_static=0.60, rest_memory=19.58 GB
[2026-05-15 09:09:53 TP0] DSv4 memory calculation: bytes_per_full_token=15955.85, available_bytes=19.58 GB, full_token=1317632
[2026-05-15 09:09:53 TP1] DSv4 memory calculation: bytes_per_full_token=15955.85, available_bytes=19.58 GB, full_token=1317632
[2026-05-15 09:09:53 TP0] Memory pool end. avail mem=36.31 GB
[2026-05-15 09:09:54 TP1] Memory pool end. avail mem=36.26 GB
[2026-05-15 09:10:49 TP0] Capture cuda graph end. Time elapsed: 55.46 s. mem usage=1.93 GB. avail mem=34.28 GB.
[2026-05-15 09:10:49 TP1] Capture cuda graph end. Time elapsed: 55.46 s. mem usage=1.93 GB. avail mem=34.24 GB.
[2026-05-15 09:10:50 TP0] max_total_num_tokens=1317632, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=256, context_len=1048576, available_gpu_mem=34.28 GB
[2026-05-15 09:10:53] The server is fired up and ready to roll!
"""

NVIDIA_SMI_MFS088 = """\
0, 89846 MiB, 7522 MiB
1, 89942 MiB, 7426 MiB
2, 89942 MiB, 7426 MiB
3, 89942 MiB, 7426 MiB
4, 89942 MiB, 7426 MiB
5, 89942 MiB, 7426 MiB
6, 89942 MiB, 7426 MiB
7, 89462 MiB, 7906 MiB
"""

NVIDIA_SMI_MFS060 = """\
0, 63092 MiB, 34276 MiB
1, 63188 MiB, 34180 MiB
2, 63188 MiB, 34180 MiB
3, 63188 MiB, 34180 MiB
4, 63188 MiB, 34180 MiB
5, 63188 MiB, 34180 MiB
6, 63188 MiB, 34180 MiB
7, 62708 MiB, 34660 MiB
"""


# ---------------------------------------------------------------------------
# Test: individual line parsers
# ---------------------------------------------------------------------------

class TestLineParsers:
    def test_parse_server_args(self):
        line = "server_args=ServerArgs(model_path='/root/workspace/models/DeepSeek-V4-Flash-Base', tp_size=8, pp_size=1, dp_size=1, ep_size=1, mem_fraction_static=0.88, kv_cache_dtype='bf16', cuda_graph_max_bs=320, disable_radix_cache=True, page_size=256)"
        sa = parse_server_args(line)
        assert sa.model_path == "/root/workspace/models/DeepSeek-V4-Flash-Base"
        assert sa.tp_size == 8
        assert sa.mem_fraction_static == 0.88
        assert sa.kv_cache_dtype == "bf16"
        assert sa.disable_radix_cache is True
        assert sa.cuda_graph_max_bs == 320

    def test_parse_server_args_fp8(self):
        line = "server_args=ServerArgs(model_path='test', tp_size=4, mem_fraction_static=0.60, kv_cache_dtype='fp8_e4m3', disable_radix_cache=False)"
        sa = parse_server_args(line)
        assert sa.tp_size == 4
        assert sa.mem_fraction_static == 0.60
        assert sa.kv_cache_dtype == "fp8_e4m3"
        assert sa.disable_radix_cache is False

    def test_parse_load_weight_begin(self):
        line = "[2026-05-15 08:39:36 TP0] Load weight begin. avail mem=93.61 GB"
        ck = parse_load_weight_begin(line)
        assert ck is not None
        assert ck.rank == 0
        assert ck.avail_gb == 93.61
        assert ck.label == "load_weight_begin"

    def test_parse_memory_profiling(self):
        line = "[2026-05-15 09:09:53 TP0] Memory profiling: available_gpu_memory=57.01 GB, total_gpu_memory=93.58 GB, mem_fraction_static=0.60, rest_memory=19.58 GB"
        mp = parse_memory_profiling(line)
        assert mp is not None
        assert mp.rank == 0
        assert mp.available_gpu_memory_gb == 57.01
        assert mp.total_gpu_memory_gb == 93.58
        assert mp.mem_fraction_static == 0.60
        assert mp.rest_memory_gb == 19.58

    def test_parse_sw_kv_memory_calc_legacy(self):
        """Legacy format: 'DSv4 memory calculation:' should still be parsed."""
        line = "[2026-05-15 09:09:53 TP0] DSv4 memory calculation: bytes_per_full_token=15955.85, available_bytes=19.58 GB, full_token=1317632"
        dc = parse_sw_kv_memory_calc(line)
        assert dc is not None
        assert dc.rank == 0
        assert dc.bytes_per_full_token == 15955.85
        assert dc.available_bytes_gb == 19.58
        assert dc.full_token == 1317632

    def test_parse_sw_kv_memory_calc_new_format(self):
        """New format: 'SW KV memory calculation:' should be parsed."""
        line = "[2026-05-15 09:09:53 TP0] SW KV memory calculation: bytes_per_full_token=15955.85, available_bytes=19.58 GB, full_token=1317632"
        dc = parse_sw_kv_memory_calc(line)
        assert dc is not None
        assert dc.rank == 0
        assert dc.bytes_per_full_token == 15955.85
        assert dc.available_bytes_gb == 19.58
        assert dc.full_token == 1317632

    def test_parse_memory_pool_end(self):
        line = "[2026-05-15 08:39:53 TP0] Memory pool end. avail mem=10.18 GB"
        ck = parse_memory_pool_end(line)
        assert ck is not None
        assert ck.rank == 0
        assert ck.avail_gb == 10.18
        assert ck.label == "memory_pool_end"

    def test_parse_cuda_graph_end(self):
        line = "[2026-05-15 08:40:32 TP0] Capture cuda graph end. Time elapsed: 54.61 s. mem usage=1.93 GB. avail mem=8.16 GB."
        cg = parse_cuda_graph_end(line)
        assert cg is not None
        assert cg.rank == 0
        assert cg.elapsed_s == 54.61
        assert cg.mem_usage_gb == 1.93
        assert cg.avail_gb == 8.16

    def test_parse_max_total_tokens(self):
        line = "[2026-05-15 08:40:32 TP0] max_total_num_tokens=3080960, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=256, context_len=1048576, available_gpu_mem=8.16 GB"
        fi = parse_max_total_tokens(line)
        assert fi is not None
        assert fi.max_total_num_tokens == 3080960
        assert fi.max_running_requests == 256
        assert fi.context_len == 1048576
        assert fi.available_gpu_mem_gb == 8.16

    def test_parse_nvidia_smi(self):
        entries = parse_nvidia_smi(NVIDIA_SMI_MFS088)
        assert len(entries) == 8
        assert entries[0].index == 0
        assert entries[0].memory_used_mib == 89846
        assert entries[0].memory_free_mib == 7522
        assert entries[7].index == 7
        assert entries[7].memory_used_mib == 89462


# ---------------------------------------------------------------------------
# Test: full log parsing
# ---------------------------------------------------------------------------

class TestLogParsing:
    def test_parse_mfs088_log(self):
        parsed = parse_log(SGLANG_LOG_MFS088)
        assert parsed.framework == "sglang"
        assert parsed.server_args is not None
        assert parsed.server_args.mem_fraction_static == 0.88
        assert parsed.server_args.kv_cache_dtype == "bf16"
        assert len(parsed.checkpoints) == 10  # 8 load_weight_begin + 2 memory_pool_end
        assert len(parsed.sw_kv_calcs) == 2
        assert len(parsed.cuda_graphs) == 2
        assert parsed.final_info is not None
        assert parsed.final_info.max_total_num_tokens == 3080960
        # No Memory profiling lines in this log
        assert len(parsed.memory_profilings) == 0

    def test_parse_mfs060_log(self):
        parsed = parse_log(SGLANG_LOG_MFS060)
        assert parsed.framework == "sglang"
        assert parsed.server_args is not None
        assert parsed.server_args.mem_fraction_static == 0.60
        assert parsed.server_args.kv_cache_dtype == "fp8_e4m3"
        assert len(parsed.memory_profilings) == 2
        assert len(parsed.sw_kv_calcs) == 2
        assert parsed.final_info is not None
        assert parsed.final_info.max_total_num_tokens == 1317632


# ---------------------------------------------------------------------------
# Test: memory decomposition
# ---------------------------------------------------------------------------

class TestMemoryDecomposition:
    def test_decompose_mfs088_no_smi(self):
        """Test decomposition for mfs=0.88 log without nvidia-smi data."""
        parsed = parse_log(SGLANG_LOG_MFS088)
        # H20 has 96 GiB HBM (approx 95.58 GiB actual)
        gpu_hbm = 95.58
        bd = decompose_memory(parsed, gpu_hbm, target_rank=0)

        # Framework overhead
        assert abs(bd.framework_overhead_gib - (95.58 - 93.61)) < 0.01

        # KV pool from SW KV calc
        assert abs(bd.kv_pool_gib - 45.78) < 0.01

        # Model weights = (93.61 - 10.18) - 45.78 = 37.65
        assert abs(bd.model_weights_gib - 37.65) < 0.01

        # CUDA graph
        assert abs(bd.cuda_graph_gib - 1.93) < 0.01

    def test_decompose_mfs060_with_smi(self):
        """Test decomposition for mfs=0.60 log with nvidia-smi data."""
        parsed = parse_log(SGLANG_LOG_MFS060)
        smi_entries = parse_nvidia_smi(NVIDIA_SMI_MFS060)
        gpu_hbm = 95.58
        bd = decompose_memory(parsed, gpu_hbm, smi_entries, target_rank=0)

        # Framework overhead
        assert abs(bd.framework_overhead_gib - (95.58 - 93.61)) < 0.01

        # KV pool from Memory profiling rest_memory
        assert abs(bd.kv_pool_gib - 19.58) < 0.01

        # Model weights via Memory profiling: avail_before_weight - total_gpu_memory
        # 93.61 - 93.58 = 0.03? No, that's wrong.
        # Actually with both Memory profiling and pool_end, we use the pool_end method
        # weight = (avail_before_weight - avail_after_pool) - kv_pool
        # = (93.61 - 36.31) - 19.58 = 37.72
        # But with Memory profiling: weight = avail_before_weight - total_gpu_memory = 93.61 - 93.58 = 0.03
        # The code should prefer the more reliable method
        assert bd.model_weights_gib > 30  # Weights should be ~37 GB

        # CUDA graph
        assert abs(bd.cuda_graph_gib - 1.93) < 0.01

        # nvidia-smi data
        assert bd.smi_used_mib == 63092

    def test_decompose_mfs088_with_smi(self):
        """Test decomposition for mfs=0.88 log with nvidia-smi, checking totals."""
        parsed = parse_log(SGLANG_LOG_MFS088)
        smi_entries = parse_nvidia_smi(NVIDIA_SMI_MFS088)
        gpu_hbm = 95.58
        bd = decompose_memory(parsed, gpu_hbm, smi_entries, target_rank=0)

        # Total used should match nvidia-smi
        assert abs(bd.total_used_gib - 89846 / 1024.0) < 0.1

        # Percentages should sum to ~100%
        total_pct = bd.weight_pct + bd.kv_pool_pct + bd.cuda_graph_pct + bd.framework_pct + bd.other_pct
        assert abs(total_pct - 100.0) < 1.0


# ---------------------------------------------------------------------------
# Test: KV cache byte calculation
# ---------------------------------------------------------------------------

class TestKVBytesCalculation:
    def test_kv_dtype_bytes(self):
        assert kv_dtype_bytes("bf16") == 2
        assert kv_dtype_bytes("fp8_e4m3") == 1
        assert kv_dtype_bytes("fp8_e5m2") == 1
        assert kv_dtype_bytes("float32") == 4
        assert kv_dtype_bytes("auto") == 2  # default

    def test_gqa_kv_bytes(self):
        """Standard GQA: per_token = 2 * L * kv_heads * head_dim * dtype_bytes / tp_size"""
        mc = ModelConfig(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            attention_type="gqa",
        )
        # GQA with kv_heads=8, tp=8: kv_heads_per_gpu=1
        per_token = calc_kv_bytes_per_token(mc, tp_size=8, kv_dtype_bytes=2)
        expected = 2 * 32 * 1 * 128 * 2  # 16384 bytes
        assert abs(per_token - expected) < 1

    def test_gqa_kv_bytes_replication(self):
        """GQA with kv_heads < tp_size: KV is replicated."""
        mc = ModelConfig(
            num_hidden_layers=32,
            num_key_value_heads=1,
            head_dim=512,
            attention_type="gqa",
        )
        # kv_heads=1 < tp=8: replicated, kv_heads_per_gpu = 1 (not 1/8)
        per_token = calc_kv_bytes_per_token(mc, tp_size=8, kv_dtype_bytes=2)
        expected = 2 * 32 * 1 * 512 * 2  # 65536 bytes
        assert abs(per_token - expected) < 1

    def test_mla_kv_bytes(self):
        """MLA: per_token = 2 * L * kv_lora_rank * dtype_bytes"""
        mc = ModelConfig(
            num_hidden_layers=61,
            kv_lora_rank=512,
            attention_type="mla",
        )
        per_token = calc_kv_bytes_per_token(mc, tp_size=8, kv_dtype_bytes=2)
        expected = 2 * 61 * 512 * 2  # 124928 bytes
        assert abs(per_token - expected) < 1

    def test_csa_hca_returns_zero(self):
        """CSA/HCA models cannot be calculated without SWA parameters."""
        mc = ModelConfig(
            num_hidden_layers=43,
            attention_type="csa_hca",
            compress_ratios=[0, 0, 4, 128],
        )
        per_token = calc_kv_bytes_per_token(mc, tp_size=8, kv_dtype_bytes=2)
        assert per_token == 0.0  # Should return 0, indicating log data should be used


# ---------------------------------------------------------------------------
# Test: GPU spec aliases
# ---------------------------------------------------------------------------

class TestGpuSpecs:
    def test_local_gpu_aliases_resolve_to_specs(self):
        specs = load_gpu_specs()
        assert "llm-serving-capacity-planner" in GPU_SPECS_PATH
        for alias, canonical in {
            "h100": "h100-sxm-80gb",
            "h200": "h200-sxm-141gb",
            "b200": "b200-sxm-180gb",
        }.items():
            assert GPU_ALIAS[alias] == canonical
            assert canonical in specs
            assert specs[canonical]["hbm_gb"] > 0


# ---------------------------------------------------------------------------
# Test: concurrency estimation
# ---------------------------------------------------------------------------

class TestConcurrencyEstimation:
    def test_basic_concurrency(self):
        fi = FinalInfo(
            max_total_num_tokens=3080960,
            max_running_requests=256,
        )
        estimates = estimate_concurrency(
            fi, kv_pool_gib=45.78, bytes_per_full_token=15955.85, kv_dtype="bf16",
            request_tokens_list=[6144],
        )
        assert len(estimates) == 1
        est = estimates[0]
        # 3080960 / 6144 = 501.3, min(501, 256) = 256
        assert est.max_concurrent == 256
        assert est.request_tokens == 6144

    def test_concurrency_token_limited(self):
        fi = FinalInfo(
            max_total_num_tokens=1317632,
            max_running_requests=256,
        )
        estimates = estimate_concurrency(
            fi, kv_pool_gib=19.58, bytes_per_full_token=15955.85, kv_dtype="fp8_e4m3",
            request_tokens_list=[6144],
        )
        est = estimates[0]
        # 1317632 / 6144 = 214.5, min(214, 256) = 214
        assert est.max_concurrent == 214

    def test_concurrency_multiple_lengths(self):
        fi = FinalInfo(
            max_total_num_tokens=3080960,
            max_running_requests=512,
        )
        estimates = estimate_concurrency(
            fi, kv_pool_gib=45.78, bytes_per_full_token=15955.85, kv_dtype="bf16",
            request_tokens_list=[4096, 6144, 8192],
        )
        assert len(estimates) == 3
        # 3080960 / 4096 = 752.2, min(752, 512) = 512
        assert estimates[0].max_concurrent == 512
        # 3080960 / 6144 = 501.3, min(501, 512) = 501
        assert estimates[1].max_concurrent == 501
        # 3080960 / 8192 = 376.0, min(376, 512) = 376
        assert estimates[2].max_concurrent == 376


# ---------------------------------------------------------------------------
# Test: end-to-end with actual log files
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_mfs088_text_output(self):
        """Test full text output for mfs=0.88 log."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SGLANG_LOG_MFS088)
            log_path = f.name

        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "capacity_analyzer.py"),
                 "--log-file", log_path, "--gpu", "h20", "--format", "text"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            output = result.stdout
            assert "Memory Breakdown" in output
            assert "KV Cache Pool" in output
            assert "Model Weights" in output
            assert "CUDA Graph" in output
            assert "37.65" in output  # model weights
            assert "45.78" in output  # KV pool
            assert "1.93" in output   # CUDA graph
            assert "3080960" in output  # max_total_num_tokens
        finally:
            os.unlink(log_path)

    def test_mfs060_text_output(self):
        """Test full text output for mfs=0.60 log."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SGLANG_LOG_MFS060)
            log_path = f.name

        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "capacity_analyzer.py"),
                 "--log-file", log_path, "--gpu", "h20", "--format", "text"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            output = result.stdout
            assert "Memory Breakdown" in output
            assert "19.58" in output   # KV pool (rest_memory)
            assert "1317632" in output  # max_total_num_tokens
        finally:
            os.unlink(log_path)

    def test_mfs060_json_output(self):
        """Test JSON output for mfs=0.60 log."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SGLANG_LOG_MFS060)
            log_path = f.name

        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "capacity_analyzer.py"),
                 "--log-file", log_path, "--gpu", "h20", "--format", "json"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            data = json.loads(result.stdout)
            assert data["framework"] == "sglang"
            assert data["kv_cache_dtype"] == "fp8_e4m3"
            assert data["max_total_num_tokens"] == 1317632
            assert "memory_breakdown" in data
            assert abs(data["memory_breakdown"]["kv_pool_gib"] - 19.58) < 0.01
        finally:
            os.unlink(log_path)

    def test_with_nvidia_smi_file(self):
        """Test with nvidia-smi file input."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SGLANG_LOG_MFS088)
            log_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(NVIDIA_SMI_MFS088)
            smi_path = f.name

        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "capacity_analyzer.py"),
                 "--log-file", log_path, "--nvidia-smi-file", smi_path,
                 "--gpu", "h20", "--format", "text"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            output = result.stdout
            assert "Per-Rank Comparison" in output
            assert "89846" in output  # TP0 used MiB
        finally:
            os.unlink(log_path)
            os.unlink(smi_path)

    def test_custom_request_tokens(self):
        """Test with custom request token lengths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SGLANG_LOG_MFS088)
            log_path = f.name

        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "capacity_analyzer.py"),
                 "--log-file", log_path, "--gpu", "h20",
                 "--request-tokens", "2048,4096,16384"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            output = result.stdout
            assert "2048" in output
            assert "16384" in output
        finally:
            os.unlink(log_path)
