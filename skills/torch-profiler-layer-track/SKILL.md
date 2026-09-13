---
name: torch-profiler-layer-track
description: Add verified layer-number guides and compact GPU display lanes to an existing Torch Profiler Chrome JSON trace. Use for L0/L1/... navigation or traces with hundreds of CUDA Graph stream rows. Changes visualization, not runtime streams.
---

# Torch Profiler Layer Track

Create a normal Chrome JSON trace with `L0, L1, ...` guides beside at most ten
synthetic GPU activity lanes per device. Keep the source file. Compaction
changes event placement, not execution: original kernel names, timestamps,
durations and `args.stream` remain intact. CPU events stay unchanged. Label
the output as a **compact view**, not a runtime stream optimization.

## Establish what the labels mean

Use the user's existing local trace, including a Desktop copy when requested.
Obtain the layer count from matching model config/source. Identify the rank/GPU,
execution phase, and one complete forward pass before counting. In speculative
decoding, separate target verify from draft passes. BS=1 does not imply one
token per target verify. A repeated block in the overview can be a whole
iteration, not a transformer layer.

Choose a GPU kernel that occurs **exactly once per layer in the selected
phase**. Verify its call site and cross-check the count with another landmark
or CPU module scopes. Record this evidence and code revision. Names, stream
numbers and layer counts depend on the model/build; do not count all Top-K
calls or blindly reuse a previous model's anchor. Inspect all GPU streams in
the rank: layers can change streams. CPU scopes mark host launch intervals,
not GPU execution boundaries.

The helper does not guess L0 from counts alone. `--anchor-offset` must identify
a verified first layer in timestamp-sorted matching events. Divisibility does
not rule out a partial first pass or mixed draft/target anchors. If evidence
is insufficient, explain the ambiguity instead of publishing confident labels.

## Add the auxiliary track

Scripts require Python 3.10+ and the standard library only. Commands are
relative to this skill directory.

```bash
python3 scripts/add_layer_track.py \
  --trace /path/to/TP-0.trace.json.gz \
  --output /path/to/TP-0.layers.trace.json \
  --anchor-regex '^YOUR_ONCE_PER_LAYER_KERNEL$' \
  --num-layers 40 --anchor-offset 0 --passes 1 \
  --max-gpu-lanes 10 \
  --phase target-verify \
  --evidence 'Config/source revision ...; first L0 verified; 40 anchors per target pass; draft uses a different path.'
```

- Use `--pid GPU_PID` when anchors occur in multiple processes/GPUs.
  Use `--device DEVICE` if a single PID contains anchors from multiple devices.
- `--passes` labels only that many complete passes. `--first-layer` supports
  a known pipeline partition. A truncated selection is rejected.
- Optional `--end-anchor-regex` identifies a verified layer-ending kernel
  occurring once between anchors and closes the final label of each pass.
  Without it, the last label is a zero-duration marker; L39 is never extended
  across a draft pass or an inter-iteration gap.
- `.json` and `.json.gz` input/output work. Output must be a new file. A mapping
  report is saved as `<output>.layers.json`.
- Use `--max-gpu-lanes 10` by default for a compact view; omit it when the user
  wants the original stream layout. The helper assigns non-overlapping activity
  to reusable display lanes and remaps matching GPU flow endpoints. It refuses
  selections needing more than ten lanes, rather than hiding concurrent work.
  Other unsupported or ambiguous events also fail with an explanation.
- Original GPU `pid`/`tid` and source event index are retained in
  `args._compact_gpu_track`; retired stream-name metadata is in the report.
  GPU user scopes are preserved in a separate `GPU scopes` process so they do
  not recreate hundreds of rows beside the kernel lanes. CPU scopes and flows
  stay in their original process.

For a trace that already has **verified guides beside the GPU**, compact it
without adding duplicate labels:

```bash
python3 scripts/compact_gpu_tracks.py \
  --trace /path/to/TP-0.layers.trace.json \
  --output /path/to/TP-0.compact-layers.trace.json \
  --max-gpu-lanes 10
```

This saves `<output>.compact.json`. Existing guides are preserved. If a guide
is missing, under the wrong process, or has unverified boundaries, regenerate
it from the original trace with `add_layer_track.py` first. Never treat the
number of synthetic lanes as the number of real CUDA streams. The original
trace remains necessary for stream scheduling and synchronization analysis.

Every interval is a **guide**, from this layer's anchor to the next layer's
anchor. Work before the anchor is outside its guide. The last interval, when
supplied, ends at a terminal kernel's completion. These are not full layer
boundaries, exclusive kernel ownership, or per-layer latency measurements.
Overlapping streams can cross labels. Numbering is zero-based: **L2 is the
third transformer layer**.
Only auxiliary endpoints are rounded to the nearest microsecond to avoid
floating-point overlaps; original timestamps and report anchor values stay intact.

## Open and inspect

Open the annotated file with Perfetto's **Open trace file**. Expand the selected
GPU/process, locate `Layer guide (...) — anchor intervals`, and pin it beside
the `GPU lane N (synthetic; ...)` tracks. Use relative times in the report to locate L0, L1, a middle
layer and the final layer.

If the browser cannot fetch a local trace URL, use the loopback viewer. It
transfers local bytes after Perfetto's PING/PONG handshake:

```bash
python3 scripts/serve_local_trace.py --trace /path/to/TP-0.layers.trace.json
# Open the printed http://127.0.0.1:PORT/ in the user's chosen browser.
# macOS example:
open -a Firefox http://127.0.0.1:PORT/
```

The server exposes only this trace and its viewer until Ctrl-C. It needs network
access for the Perfetto UI; it does not upload the trace to storage. Manual
file opening works when the UI is locally available. See
[navigation.md](references/navigation.md) for SQL checks, screenshot guidance,
time units, and the DeepSeek-V4.1 example.

## Verify and deliver

1. Check the report's PID, layer count, anchor offset, phase and source hash
   against the intended trace. Check two passes when claiming the mapping
   repeats across iterations.
2. Verify that reversing the recorded placement changes and restoring retired
   metadata reconstructs the source event array exactly. Without compaction,
   the source array remains an unchanged prefix. Check the original file hash.
   In Perfetto, check both kernel count and actual imported track count: every
   GPU activity lane and layer guide should have depth zero. A raw JSON tid
   count alone can miss additional overflow tracks created by the importer.
3. Inspect the guide beside the GPU kernels. If requested, capture the real
   viewer with readable labels and relevant streams. Preserve CPU tracks/flows
   when present; labels cannot reconstruct CPU events absent from the capture.
4. Return the compact trace, report, and screenshot if requested. State the
   original stream count, synthetic lane count, numbering convention and
   anchor-boundary limitation. Keep large traces and
   private capture paths out of public skill/PR artifacts.
