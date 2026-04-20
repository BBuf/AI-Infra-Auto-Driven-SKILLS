## Worked Example: Upstream Top-K Corruption, Downstream Shared-Memory OOB

Use this case study when you want an intentional CUDA incident that:

- crashes in a downstream MoE align kernel
- looks like a shared-memory out-of-bounds or illegal-address issue
- actually originates in the previous routing kernel
- is best reproduced through crash dump plus replay rather than ad-hoc prompts
- behaves like a production-serving incident instead of a toy standalone kernel crash

This is not meant to replace a low-level CUDA graph or single-kernel debug
playbook. The point of this example is different: the symptom emerges in a real
serving path, depends on request shape, surfaces first as a generic runtime
failure, and only becomes obvious after going through the full
crash-dump-to-replay-to-coredump workflow.

## Target Path

Use a Qwen3 MoE model that routes through `fused_topk` instead of grouped top-k:

- model: `Qwen/Qwen3-30B-A3B`
- `num_experts=128`
- `num_experts_per_tok=8`
- `use_grouped_topk=False`

The important call chain is:

1. [python/sglang/srt/models/qwen3_moe.py](../../../../sglang/python/sglang/srt/models/qwen3_moe.py)
2. [python/sglang/srt/layers/moe/topk.py](../../../../sglang/python/sglang/srt/layers/moe/topk.py)
3. [sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu](../../../../sglang/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu)
4. [sgl-kernel/csrc/moe/moe_align_kernel.cu](../../../../sglang/sgl-kernel/csrc/moe/moe_align_kernel.cu)

For this model shape, `topk_softmax` dispatches to `topkGatingSoftmax`, not the `moeTopKFast` fallback.

## Why This Incident Is Useful

The visible crash shows up in `moe_align_block_size_kernel` here:

```cpp
int expert_id = topk_ids[i] + 1;
atomicAdd(&shared_counts[expert_id], 1);
```

If one `topk_ids[i]` is corrupted to a very large positive value, the `atomicAdd`
lands far past `shared_counts[num_experts]`. In `cuda-gdb`, this makes the failing
kernel look like the align kernel itself is wrong, even though the bad value was
written one kernel earlier.

That is the exact behavior you want for an incident-triage skill:

- crash dump identifies the triggering request shape
- replay makes the crash repeatable
- CUDA coredump tells you which kernel actually faulted
- code reading is still needed to walk one step upstream and find the real source

## Injection Site

Patch the producer, not the consumer.

Use this site in
[sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu](../../../../sglang/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu)
inside `topkGatingSoftmax`, immediately after the normal `indices[idx]` write:

```cpp
if (thread_group_idx == 0) {
  const bool node_uses_expert = expert >= start_expert && expert < end_expert;
  const bool should_process_row = row_is_active && node_uses_expert;

  const int idx = k * thread_row + k_idx;
  output[idx] = max_val;
  indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;

  if (should_process_row && NUM_EXPERTS == 128 && k == 8 && num_rows == 769 &&
      thread_row == 17 && k_idx == 0) {
    indices[idx] = NUM_EXPERTS + 4096;
  }

  row_sum_for_renormalize += max_val;
}
```

Design notes:

- Keep the corruption in the producer kernel. Do not modify `moe_align_block_size_kernel`.
- Corrupt exactly one slot. More corruption makes the root cause too obvious.
- Use a large positive invalid value such as `NUM_EXPERTS + 4096`. Small overruns can just scribble into nearby shared-memory regions and create ambiguous secondary symptoms.
- Do not add `printf`, `assert`, or metric logging in the injected path.
- Make the guard depend on request shape. `num_rows == 769` is only an example magic value; any stable rare prefill shape is acceptable.

## Trigger Shape

One validated trigger prompt was:

```text
"hello " * 768
```

On `Qwen/Qwen3-30B-A3B`, that prompt tokenizes to `769` prompt tokens. With the
injected producer corruption, the bad routing entry becomes:

```text
topk_ids[17, 0] = 4224
```

That value is far outside the valid expert-id range `[0, 127]`, but the visible
failure still lands later in `moe_align_block_size_kernel`.

Treat that prompt as only one concrete instance. The transferable part is the
shape-sensitive guard in the producer kernel, not the exact hardware or host.

## Production-Oriented Reproduction Flow

### 1. Patch the producer in the serving build

Apply the injection only in:

```bash
<sglang-root>/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
```

Do not touch:

- `<sglang-root>/sgl-kernel/csrc/moe/moe_align_kernel.cu`

That preserves the downstream false lead.

### 2. Rebuild and install `sglang-kernel`

```bash
cd <sglang-root>/sgl-kernel
make build
```

Install the rebuilt package into the exact serving environment that will be used
for replay. A direct kernel sanity check should confirm that the producer really
emits an invalid top-k index:

```bash
row17_col0 4224
max_idx 4224
```

### 3. Launch the incident server and collect a crash dump

Keep the launch close to the real serving path. The exact topology is not the
important part; preserving the same model path, TP setting, CUDA-graph flags,
and request-dump behavior is.

```bash
export CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1>
export PYTHONPATH=<sglang-root>/python
cd <sglang-root>/python
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B \
  --tp 2 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --crash-dump-folder <crash-dump-folder>
```

Let production-like traffic or a request generator hit the server until the
incident occurs and a dump is written. Do not hand-tune prompts first if your
goal is to practice the production triage path.

### 4. Summarize the crash dump instead of guessing the prompt shape

In one validated run, the dump contained two requests:

- request `[0]`: short warmup prompt
- request `[1]`: the long `"hello " * 768` trigger prompt

```bash
python3 scripts/summarize_request_dump.py \
  --input-file <crash-dump-file>
```

### 5. Replay the captured request mix

Use the trusted local helper if stock replay is blocked by `safe_pickle_load`:

```bash
python3 scripts/replay_trusted_request_dump.py \
  --input-file <crash-dump-file> \
  --host 127.0.0.1 \
  --port 32000 \
  --parallel 1
```

In the first runtime-injection smoke run, the client-side visible symptom was
only:

```text
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

That is the kind of misleading outer symptom this example is meant to teach.

### 6. Restart with CUDA coredumps and replay again

```bash
export CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1>
export PYTHONPATH=<sglang-root>/python
export SGLANG_CUDA_COREDUMP=1
export SGLANG_CUDA_COREDUMP_DIR=<coredump-folder>
cd <sglang-root>/python
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B \
  --tp 2 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --crash-dump-folder <crash-dump-folder>
```

Replay the same dump again. Then inspect the generated coredump:

```bash
cuda-gdb "$(which python3)" \
  -ex "set pagination off" \
  -ex "target cudacore <coredump-file>" \
  -ex "where" \
  -ex "info cuda kernels" \
  -ex "x/12i <faulting-pc>"
```

The artifact pattern usually looks like:

- `<crash-dump-folder>/<worker-id>/crash_dump_<timestamp>.pkl`
- `<coredump-folder>/cuda_coredump_<host>.<pid>.<ts>`

## Expected Triage Result

On one validated run, `cuda-gdb` reported:

```text
CUDA Exception: Warp Out-of-range Address
The exception was triggered at PC 0x7f7fe1dfac70  void moe_align_block_size_kernel<int>(...)
#0  0x00007f7fe1dfac00 in void moe_align_block_size_kernel<int>(...)<<<(2,1,1),(1024,1,1)>>> ()
```

The first SASS instructions around the faulting PC were:

```text
*> 0x7f7fe1dfac70 <...+624>: ATOMS.POPC.INC.32 RZ[R10+URZ+0x4]
   0x7f7fe1dfac80 <...+640>: @!P0 BRA 0x7f7fe1dfabf0
   0x7f7fe1dfac90 <...+656>: BSYNC B0
   0x7f7fe1dfaca0 <...+672>: BAR.SYNC.DEFER_BLOCKING 0x0
   0x7f7fe1dfacb0 <...+688>: ISETP.NE.AND P0,PT,R17,RZ,PT
   0x7f7fe1dfacc0 <...+704>: BSSY B0,0x7f7fe1dfae90
   0x7f7fe1dfacd0 <...+720>: @P0 BRA 0x7f7fe1dfae80
   0x7f7fe1dface0 <...+736>: LDC R10,c[0x0][0x234]
   0x7f7fe1dfacf0 <...+752>: LDS R11[R12]
```

If the incident is behaving as intended, you should see this progression:

1. The process crashes only for a narrow request shape.
2. The crash dump preserves the exact launch command and request mix.
3. Replay reproduces the crash without manual prompt fiddling.
4. `cuda-gdb` points at `moe_align_block_size_kernel`.
5. The faulting instruction is near the shared-memory `atomicAdd`.
6. Reading one kernel upstream shows that `topk_ids` was already corrupted in `topkGatingSoftmax`.

The important conclusion is:

- failing kernel: `moe_align_block_size_kernel`
- root-cause kernel: `topkGatingSoftmax`

## What Not To Do

- Do not add a bounds check in `moe_align_block_size_kernel` first and call it fixed.
- Do not trust the faulting PC as proof that the consumer kernel is the origin.
- Do not swap to another model path that uses grouped top-k. That bypasses this example.
- Do not skip replay. Without replay, you are back to guessing at prompt shape.
- Do not reduce this to a one-off kernel debug exercise. The important lesson is
  the production incident flow: collect, replay, coredump, then walk upstream.

## Optional Follow-Up

After demonstrating the incident, fix it by removing the injected corruption and rerun the same replay.
The replay should stop crashing without any changes to `moe_align_block_size_kernel`.
