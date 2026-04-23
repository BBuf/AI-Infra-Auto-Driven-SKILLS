---
name: h100
description: SSH into host `h100_sglang`, enter Docker container `sglang_bbuf`, work in `/data/bbuf/repos/sglang`, and use the ready H100 remote environment for SGLang development and validation. Use when a task needs remote CUDA work, GPU-backed smoke tests, vLLM / TensorRT-LLM bring-up, diffusion checks, or a safe remote copy instead of local-only execution.
---

# H100

## Overview

Use this skill to do SGLang development on the H100 box through `h100_sglang`.
The default container is `sglang_bbuf` and the repo lives at `/data/bbuf/repos/sglang`.

For transient validation state, copied worktrees, logs, profiler traces, and container
side outputs, use `/data/bbuf/validate/<task>`.
Do not write under `/home`.

On this host, direct host-user writes to `/data/bbuf/...` are not always reliable.
Create and mutate those paths through `docker exec sglang_bbuf ...` or from inside the
framework container that owns the run.

This environment is already prepared:

- `sglang_bbuf` is running on `lmsysorg/sglang:dev`
- the repo is cloned at `/data/bbuf/repos/sglang`
- editable installs for `python[all]` and `python[diffusion]` are already done
- `/data/.cache` is mounted to `/root/.cache`
- Infiniband paths are mounted into the container for RDMA-aware workflows:
  `/sys/class/infiniband`, `/dev/infiniband`, and `/usr/sbin/show_gids`

Hugging Face cache is already mounted, but do not assume `HF_TOKEN` is visible in every
`docker exec` context. Interactive shells and non-interactive `docker exec ... bash -lc`
can behave differently. Always verify with `echo ${HF_TOKEN:+set}` before gated-model or
Hub-backed runs.

## Quick Start

1. Check the host, container, and GPU state.

```bash
ssh h100_sglang 'hostname && whoami'
ssh h100_sglang 'docker ps --format "table {{.Names}}\t{{.Status}}" | sed -n "1,20p"'
ssh h100_sglang 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

2. Enter the default container and repo.

```bash
ssh h100_sglang 'docker exec -it sglang_bbuf /bin/zsh'
cd /data/bbuf/repos/sglang
echo ${HF_TOKEN:+set}
```

If `HF_TOKEN` is missing in the current shell, export it before Hub-backed workflows:

```bash
export HF_TOKEN=<your-hf-token>
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

For non-interactive `docker exec ... bash -lc "<cmd>"` runs, prefer exporting both
variables inside the command itself instead of assuming shell startup will populate them.

3. Pick a free GPU.

Use a GPU with `0` utilization and only a few MiB allocated.
Always set `CUDA_VISIBLE_DEVICES=<gpu_id>` for GPU-backed validation commands.

4. This host does not provide an automatic `kill-idle` helper.

Do not assume you can reclaim other users' allocations automatically.
If the free GPU list is tight, re-check `nvidia-smi`, choose another GPU, or coordinate
before proceeding.

5. If the container is not running, start it first.

```bash
ssh h100_sglang 'docker start sglang_bbuf'
```

## Safe Remote Workflow

1. Inspect the default repo before editing it.

```bash
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "cd /data/bbuf/repos/sglang && git branch --show-current && git status --short"'
```

2. Fast-forward the default repo before creating any validation worktree.

```bash
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "cd /data/bbuf/repos/sglang && git fetch origin && git checkout main && git pull --ff-only origin main"'
```

3. Avoid writing directly into `/data/bbuf/repos/sglang` when it is dirty or when the
local snapshot differs from remote `HEAD`.

4. Prefer one of these isolation strategies.

Create a detached worktree for remote-only experiments:

```bash
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "mkdir -p /data/bbuf/validate && cd /data/bbuf/repos/sglang && git worktree add --detach /data/bbuf/validate/sglang_validate_h100 HEAD"'
```

Stream the exact local working tree into the container when validating the current local snapshot:

```bash
COPYFILE_DISABLE=1 tar --exclude=.git -cf - . | \
ssh h100_sglang 'docker exec -i sglang_bbuf sh -lc "rm -rf /data/bbuf/validate/sglang_local_validate && mkdir -p /data/bbuf/validate/sglang_local_validate && tar -xf - -C /data/bbuf/validate/sglang_local_validate"'
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "find /data/bbuf/validate/sglang_local_validate -name '\''._*'\'' -delete"'
```

Use the streamed copy when the goal is "validate exactly what is in the local repo right now".

For patch-oriented remote validation:

- update remote `main`
- create a detached worktree from that clean commit
- stream or apply only the focused local diff into the worktree

That keeps `/data/bbuf/repos/sglang` clean while still validating the exact local delta.

## General Validation Workflow

1. Start with import or syntax-level checks.

```bash
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "cd /data/bbuf/validate/sglang_local_validate && python -m compileall python/sglang"'
```

For diffusion-specific edits, prefer a narrower first pass:

```bash
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "cd /data/bbuf/validate/sglang_local_validate && python -m compileall python/sglang/jit_kernel/diffusion/triton python/sglang/multimodal_gen/runtime/layers"'
```

2. Run targeted tests for the changed area.

```bash
ssh h100_sglang 'docker exec sglang_bbuf env PYTHONPATH=python bash -lc "cd /data/bbuf/validate/sglang_local_validate && pytest -q path/to/test.py -q"'
```

3. For GPU-backed changes, pin a free GPU explicitly.

```bash
ssh h100_sglang 'docker exec sglang_bbuf env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python bash -lc "cd /data/bbuf/validate/sglang_local_validate && pytest -q path/to/gpu_test.py -q"'
```

4. Attempt model-level or server-level smoke only after unit, kernel, or targeted regression checks pass.

Treat checkpoint, dependency, and environment failures separately from code regressions.
If a workflow reads from Hugging Face Hub, verify `HF_TOKEN` first and re-export it in
the current shell or command when needed.

## External LLM Containers

Use the host Docker daemon, not `docker exec sglang_bbuf`, when validating other serving
stacks such as vLLM or TensorRT-LLM.

Reuse the shared host-side Hugging Face cache:

```bash
-v /data/.cache/huggingface:/root/.cache/huggingface
```

When you need logs, traces, or analysis outputs shared with `sglang_bbuf`, also mount a
`/data` validation root into the external container:

```bash
-v /data/bbuf/validate/unified_llm_profiler_skill:/data/bbuf/validate/unified_llm_profiler_skill
```

Before launching external containers, create the validation root through `sglang_bbuf`:

```bash
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "mkdir -p /data/bbuf/validate/unified_llm_profiler_skill/runs"'
```

Prefer single-card bring-up first.
Only scale to multiple GPUs if the single-card load fails or the target model clearly needs it.

Validated images on `2026-04-22`:

- `vllm/vllm-openai:latest` reporting `vLLM 0.19.1`
- `nvcr.io/nvidia/tensorrt-llm/release:latest` reporting `TensorRT-LLM 1.0.0`

### vLLM Single-GPU Template

```bash
ssh h100_sglang '
export HF_TOKEN=<your-hf-token>
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
docker rm -f vllm_qwen3_test >/dev/null 2>&1 || true
docker pull vllm/vllm-openai:latest
docker run -d --rm \
  --name vllm_qwen3_test \
  --gpus all \
  --ipc=host \
  --network host \
  -e CUDA_VISIBLE_DEVICES=7 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -v /data/.cache/huggingface:/root/.cache/huggingface \
  -v /data/bbuf/validate/unified_llm_profiler_skill:/data/bbuf/validate/unified_llm_profiler_skill \
  vllm/vllm-openai:latest \
  Qwen/Qwen3-14B \
  --host 0.0.0.0 \
  --port 31083 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/data/bbuf/validate/unified_llm_profiler_skill/runs/qwen3_14b/vllm_profile\"}"
'
```

### TensorRT-LLM Single-GPU Template

```bash
ssh h100_sglang '
export HF_TOKEN=<your-hf-token>
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
docker rm -f trt_qwen3_test >/dev/null 2>&1 || true
docker pull nvcr.io/nvidia/tensorrt-llm/release:latest
docker run -d --rm \
  --name trt_qwen3_test \
  --gpus all \
  --ipc=host \
  --network host \
  --entrypoint bash \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e TLLM_TORCH_PROFILE_TRACE=/data/bbuf/validate/unified_llm_profiler_skill/runs/qwen3_14b/trtllm_profile/trace.json \
  -v /data/.cache/huggingface:/root/.cache/huggingface \
  -v /data/bbuf/validate/unified_llm_profiler_skill:/data/bbuf/validate/unified_llm_profiler_skill \
  nvcr.io/nvidia/tensorrt-llm/release:latest \
  -lc "mkdir -p /data/bbuf/validate/unified_llm_profiler_skill/runs/qwen3_14b/trtllm_profile && \
       trtllm-serve serve Qwen/Qwen3-14B \
         --backend pytorch \
         --tp_size 1 \
         --gpus_per_node 1 \
         --host 0.0.0.0 \
         --port 32083 \
         --max_seq_len 4096 \
         --kv_cache_free_gpu_memory_fraction 0.85"
'
```

### Readiness And Mini Benchmark

```bash
ssh h100_sglang 'for p in 30083 31083 32083; do echo ---$p---; curl -sf http://127.0.0.1:$p/v1/models | head -c 300 || true; echo; done'
```

```bash
ssh h100_sglang 'PORT=31083 MODEL=Qwen/Qwen3-14B python3 - <<'"'"'PY'"'"'
import json, os, statistics, time, urllib.request
prompts = [
    "用一句中文介绍上海。",
    "What is 2+2? Answer briefly.",
    "Write one short haiku about GPUs.",
]
url = f"http://127.0.0.1:{os.environ['PORT']}/v1/chat/completions"
latencies = []
samples = []
errors = []
for i in range(6):
    payload = {
        "model": os.environ["MODEL"],
        "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
        "temperature": 0,
        "max_tokens": 64,
    }
    start = time.time()
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            body = json.loads(resp.read().decode())
        latencies.append(time.time() - start)
        samples.append(body["choices"][0]["message"].get("content", "")[:220])
    except Exception as exc:
        errors.append(repr(exc))
print(json.dumps({
    "success": len(samples),
    "errors": len(errors),
    "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else None,
    "samples": samples[:3],
    "error_samples": errors[:3],
}, ensure_ascii=False))
PY'
```

Set `PORT=30083` for SGLang or `PORT=32083` for TensorRT-LLM.

## Real H100 Validation Matrix

The latest unified LLM profiler workflow was revalidated on `2026-04-23` under:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3`

The rendered markdown bundle for the final three-model matrix is:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3/h100_large_model_matrix_v3_bundle.md`

Each lane below completed real serving bring-up, profiler capture, and live probe requests
on `4x H100`.

| Model | SGLang | vLLM | TensorRT-LLM | Notes |
| --- | --- | --- | --- | --- |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; live outputs were direct and non-empty |
| `Qwen/Qwen2.5-32B-Instruct` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; live outputs were direct and non-empty |
| `Qwen/Qwen3-32B` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; vLLM and TensorRT-LLM chat mode commonly emitted `<think>` prefixes |

Older single-card smoke still matters when the model fits:

| Model | SGLang | vLLM | TensorRT-LLM | Notes |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-8B` | `1x H100` | `1x H100` | `1x H100` | earlier single-card unified profiler path validated on all three frameworks |
| `Qwen/Qwen3-14B` | `1x H100` | `1x H100` | `1x H100` | earlier single-card unified profiler path validated on all three frameworks |
| `Qwen/Qwen3-30B-A3B` | `1x H100` in SGLang with `--mem-fraction-static 0.80` | `1x H100` | `1x H100` | earlier single-card unified profiler path validated on all three frameworks |

Notes from the current H100 run:

- keep all profiler traces, benchmark JSON, and rendered markdown bundles under `/data/...`
- if the host user cannot mutate a `/data/...` path directly, create or update it through `docker exec sglang_bbuf ...`
- prefer `1x GPU` first for smoke or cookbook replay; widen to `4x GPU` only for models that need it or for the final matrix run
- align all three frameworks on the same three-table output shape; compare `extend/prefill` and `decode` separately instead of mixing the stages
- keep SGLang kernel-site reconstruction deterministic; the current parser leaves sampling off for SGLang so optimized trace parsing does not change the table content
- pin TensorRT-LLM to `--backend pytorch`
- current workflow uses vLLM `--profiler-config {"profiler":"torch","torch_profiler_dir":"..."}` and TensorRT-LLM `TLLM_PROFILE_START_STOP` plus `TLLM_TORCH_PROFILE_TRACE`

## Cleanup

Remove temporary validation directories when finished.

```bash
ssh h100_sglang 'docker exec sglang_bbuf bash -lc "rm -rf /data/bbuf/validate/sglang_local_validate /data/bbuf/validate/sglang_validate_h100"'
```

Remove external containers explicitly when they are no longer needed.

```bash
ssh h100_sglang 'docker rm -f vllm_qwen3_test trt_qwen3_test >/dev/null 2>&1 || true'
```
