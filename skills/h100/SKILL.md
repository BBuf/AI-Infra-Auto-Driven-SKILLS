---
name: h100
description: SSH into host `h100_sglang`, enter Docker container `sglang_bbuf`, work in `/sgl-workspace/sglang`, and use the ready H100 remote environment for SGLang development and validation. Use when a task needs remote CUDA work, GPU-backed smoke tests, diffusion checks, or a safe remote copy instead of local-only execution.
---

# H100

## Overview

Use this skill to do SGLang development on the H100 box through `h100_sglang`.
The default container is `sglang_bbuf` and the repo lives at `/sgl-workspace/sglang`.
Prefer it whenever local validation is insufficient for CUDA, Triton, diffusion pipelines, or other GPU-backed SGLang behavior.

This environment is already prepared:

- `sglang_bbuf` is running on `lmsysorg/sglang:dev`
- the repo is cloned at `/sgl-workspace/sglang`
- editable installs for `python[all]` and `python[diffusion]` are already done
- `/root/.cache` is mounted as the cache path
- Infiniband paths are mounted into the container for RDMA-aware workflows:
  `/sys/class/infiniband`, `/dev/infiniband`, and `/usr/sbin/show_gids`

Hugging Face cache is already mounted, but do not assume `HF_TOKEN` is visible in
every `docker exec` context. Interactive shells and non-interactive `docker exec
... bash -lc "<cmd>"` can behave differently. Always verify with
`echo ${HF_TOKEN:+set}` before gated-model or Hub-backed runs.

## Quick Start

1. Check the host, container, and GPU state.

```bash
ssh h100_sglang 'hostname && whoami'
ssh h100_sglang 'docker ps --format "table {{.Names}}\t{{.Status}}" | sed -n "1,20p"'
ssh h100_sglang 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

2. Enter the container and repo.

```bash
ssh h100_sglang 'docker exec -it sglang_bbuf /bin/zsh'
cd /sgl-workspace/sglang
echo ${HF_TOKEN:+set}
```

If `HF_TOKEN` is unexpectedly missing in the current shell, export it manually before Hub-backed workflows:

```bash
export HF_TOKEN=<your-hf-token>
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

For non-interactive `docker exec ... bash -lc "<cmd>"` runs, prefer exporting both
variables inside the command itself instead of assuming the shell startup path
will populate them.

3. Pick a free GPU.

Use a GPU with `0` utilization and only a few MiB allocated.
Set `CUDA_VISIBLE_DEVICES=<gpu_id>` for every GPU-backed validation command.

4. This host currently does not provide the `kill-idle` helper.

Do not assume you can reclaim other users' idle allocations automatically.
If the free GPU list is tight, re-check `nvidia-smi`, choose another GPU, or coordinate before proceeding.

5. If the container is not running, start it first.

```bash
ssh h100_sglang 'docker start sglang_bbuf'
```

## Safe Remote Workflow

1. Inspect the default repo before editing it.

```bash
ssh h100_sglang 'docker exec sglang_bbuf zsh -lc "cd /sgl-workspace/sglang && git branch --show-current && git status --short"'
```

2. Fast-forward `/sgl-workspace/sglang` to the latest clean `main` before creating
any validation worktree.

```bash
ssh h100_sglang 'docker exec sglang_bbuf zsh -lc "cd /sgl-workspace/sglang && git fetch origin && git checkout main && git pull --ff-only origin main"'
```

3. Avoid writing directly into `/sgl-workspace/sglang` when it is dirty or when the local snapshot differs from the remote `HEAD`.

4. Prefer one of these isolation strategies.

Create a detached worktree for remote-only experiments:

```bash
ssh h100_sglang 'docker exec sglang_bbuf zsh -lc "cd /sgl-workspace/sglang && git worktree add --detach /tmp/sglang_validate_h100 HEAD"'
```

Stream the exact local working tree into the container when validating the current local snapshot:

```bash
COPYFILE_DISABLE=1 tar --exclude=.git -cf - . | \
  ssh h100_sglang 'docker exec -i sglang_bbuf sh -lc "rm -rf /tmp/sglang_local_validate && mkdir -p /tmp/sglang_local_validate && tar -xf - -C /tmp/sglang_local_validate"'
ssh h100_sglang 'docker exec sglang_bbuf zsh -lc "find /tmp/sglang_local_validate -name '\''._*'\'' -delete"'
```

Use the streamed copy when the goal is "validate exactly what is in the local repo right now".
For patch-oriented remote validation, another good option is:

- update remote `main`
- create a detached worktree from that clean commit
- stream or apply a focused local patch diff into the worktree only

That keeps `/sgl-workspace/sglang` clean while still validating the exact local delta.

## Validation Workflow

1. Start with import or syntax-level checks.

```bash
ssh h100_sglang 'docker exec sglang_bbuf zsh -lc "cd /tmp/sglang_local_validate && python -m compileall python/sglang"'
```

For diffusion-specific edits, prefer a narrower first pass:

```bash
ssh h100_sglang 'docker exec sglang_bbuf zsh -lc "cd /tmp/sglang_local_validate && python -m compileall python/sglang/jit_kernel/diffusion/triton python/sglang/multimodal_gen/runtime/layers"'
```

2. Run targeted tests for the changed area.

```bash
ssh h100_sglang 'docker exec sglang_bbuf env PYTHONPATH=python zsh -lc "cd /tmp/sglang_local_validate && pytest -q path/to/test.py -q"'
```

For diffusion changes, start with the fused modulation regression:

```bash
ssh h100_sglang 'docker exec sglang_bbuf env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python zsh -lc "cd /tmp/sglang_local_validate && pytest -q python/sglang/jit_kernel/tests/test_qwen_image_modulation.py -q"'
```

3. For GPU-backed changes, pin a free GPU explicitly.

```bash
ssh h100_sglang 'docker exec sglang_bbuf env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python zsh -lc "cd /tmp/sglang_local_validate && pytest -q path/to/gpu_test.py -q"'
```

4. For kernel-heavy diffusion work, run a targeted smoke script for the changed primitives before attempting a model-level run.

Cover at least these when relevant:

- `rms_norm_fn`
- `RMSNorm` under `torch.compile`
- `norm_infer`
- `apply_rotary_embedding`

Pipe the script through `docker exec -i ... python` for pure kernel smoke.

5. Use a real `.py` file with `if __name__ == "__main__":` when calling `DiffGenerator.from_pretrained(..., local_mode=True)` or any flow that relies on `multiprocessing.spawn`.

`multiprocessing.spawn` will fail if the script is executed from stdin or from unguarded top-level code.

6. Attempt model-level or server-level smoke only after unit, kernel, or targeted regression checks pass.

Treat checkpoint, dependency, and environment failures separately from code regressions.
If a workflow reads from Hugging Face Hub, verify `HF_TOKEN` first and re-export it
explicitly in the current shell or command when needed.

## Torch Compile Attribution

When a benchmark compares eager vs `torch.compile`, do not stop at the speedup number.
Capture matching eager and compile traces or perf dumps, then run `scripts/analyze_diffusion_torch_compile.py` from the repo to explain where the gain came from.

## Cleanup

Remove temporary validation directories when finished.

```bash
ssh h100_sglang 'docker exec sglang_bbuf rm -rf /tmp/sglang_local_validate /tmp/sglang_validate_h100'
```
