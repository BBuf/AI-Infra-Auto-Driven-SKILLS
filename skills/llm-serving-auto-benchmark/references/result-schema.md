# Result Schema

Write one JSON object per candidate. Keep failed candidates in the same file so
the final summary explains what was tried.

## JSONL Row

```json
{
  "framework": "sglang",
  "framework_version": "0.5.0",
  "framework_commit": "abcdef0",
  "candidate_id": "sglang-tp8-flashinfer",
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "status": "ok",
  "failure_reason": "",
  "hardware": {
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "gpu_count": 8,
    "visible_devices": "0,1,2,3,4,5,6,7"
  },
  "workload": {
    "kind": "custom",
    "scenario": "chat",
    "dataset_path": "/bench/workload.autobench.jsonl",
    "num_prompts": 1000,
    "request_rate": 16,
    "max_concurrency": 256,
    "endpoint": "/v1/chat/completions"
  },
  "sla": {
    "max_p99_ttft_ms": 2000,
    "max_p99_tpot_ms": 80,
    "min_success_rate": 0.99,
    "passed": true
  },
  "metrics": {
    "request_throughput": 15.8,
    "output_token_throughput": 12500.0,
    "total_token_throughput": 42000.0,
    "mean_ttft_ms": 430.0,
    "p99_ttft_ms": 1550.0,
    "mean_tpot_ms": 26.0,
    "p99_tpot_ms": 72.0,
    "mean_e2e_ms": 8200.0,
    "p99_e2e_ms": 19000.0,
    "success_rate": 0.995
  },
  "server_command": "python -m sglang.launch_server ...",
  "benchmark_command": "python -m sglang.bench_serving ...",
  "accuracy": {
    "mmlu": {
      "accuracy": 0.742,
      "num_examples": 14042,
      "subcategories": {
        "stem": 0.701,
        "humanities": 0.781,
        "social_sciences": 0.764,
        "other": 0.733
      },
      "command": "python -m sglang.test.run_eval --eval-name mmlu ...",
      "result_file": "/bench/sglang/accuracy/mmlu.json",
      "report": "/bench/sglang/accuracy/mmlu.html"
    },
    "gsm8k": {
      "accuracy": 0.835,
      "num_examples": 1314,
      "command": "python -m sglang.test.run_eval --eval-name gsm8k ...",
      "result_file": "/bench/sglang/accuracy/gsm8k.json"
    }
  },
  "validated_cli_flags": {
    "server": ["tp_size", "attention_backend"],
    "benchmark": ["dataset_name", "request_rate", "max_concurrency"]
  },
  "artifacts": {
    "server_log": "/bench/sglang/server.log",
    "raw_result": "/bench/sglang/results.jsonl",
    "server_help": "/bench/sglang/help_launch_server.txt",
    "benchmark_help": "/bench/sglang/help_bench_serving.txt"
  }
}
```

## Status Values

- `ok`: benchmark finished and metrics are trustworthy
- `failed`: command failed for a known non-OOM reason
- `oom`: model or candidate exhausted GPU/host memory
- `timeout`: server or benchmark timed out
- `skipped`: intentionally not run, with a reason in `failure_reason`

## Ranking Rule

The default ranking is:

1. `status == "ok"`
2. `sla.passed == true`
3. higher `metrics.request_throughput`
4. higher `metrics.output_token_throughput`
5. lower `metrics.p99_ttft_ms`
6. lower `metrics.p99_tpot_ms`
7. lower `hardware.gpu_count`

If the user cares more about token throughput than request throughput, swap
steps 3 and 4 and state that in the final report.

## Final Report Tables

The markdown summary must include these sections:

1. `Best Commands By Framework`: one table per framework. Each table has one row
   per workload scenario and includes the best candidate, SLA result, throughput,
   latency metrics, GPU count, exact server command, and artifacts.
2. `Cross-Framework Best Comparison`: one table that compares the best SGLang,
   vLLM, and TensorRT-LLM command for each scenario. Sort each scenario by the
   ranking rule above so the best deployment choice is first.
3. `Accuracy Of Selected Deployment Commands`: one table for the selected
   deployment commands. Include MMLU final accuracy, MMLU subgroup accuracy when
   available, GSM8K accuracy, evaluated example counts, and accuracy artifacts.

Accuracy rows are required for a final report. If the evaluator cannot provide
MMLU subgroup metrics for a framework, keep the final MMLU score and leave the
subgroup columns empty with a note in the artifact or caveat text.
