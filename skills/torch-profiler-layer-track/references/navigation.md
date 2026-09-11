# Layer navigation and evidence

## What to count

A model-specific, once-per-layer anchor is necessary. Check the exact build's
call site; fused kernels may change names or remove earlier anchors. A kernel
can run once per attention/FFN half, once per request, or once per draft stage.
Those are not interchangeable with one occurrence per transformer layer.

For DeepSeek-V4.1 in the examined optimized build, `_q_rope_store` occurred
40 times per target verify and 800 times over 20 iterations. Draft used another
RoPE path. The main-model MoE finalize kernel with six routed experts gave a
second 800-event landmark; draft used three routed experts. These are evidence
from one build, not built-in detection rules. Ordinary-decode, older builds,
other tensor-parallel sizes, and fused successors must be rechecked.

Example after verifying those conditions in the supplied trace:

```bash
python3 scripts/add_layer_track.py \
  --trace /path/to/optimized.trace.json \
  --output /path/to/optimized.layers.trace.json \
  --anchor-regex '^_q_rope_store$' \
  --end-anchor-regex 'moe_finalize_all_reduce_kernel<4u, 5120u, 6u' \
  --num-layers 40 --anchor-offset 0 --passes 20 \
  --phase target-verify \
  --evidence 'Matching source confirms one Q-store per target layer; 800 Q stores and 800 main MoE finalizers; first anchor verified as L0; draft uses a distinct path.'
```

The finalizer regex is intentionally build-specific. Terminal validation
requires exactly one match between each pair of anchors. A mismatch must be
investigated, not bypassed by shortening the model layer count until it fits.
For a partial capture, locate the first complete pass and set the offset.
The report's `pass_index` is relative to the selected subset, not necessarily
an absolute scheduler iteration number.

## Interpreting the track

`L0` is the first transformer layer; `L20` is the twenty-first. A guide spans
`anchor(Li)` to `anchor(Li+1)`. It may contain later work from Li and work from
Li+1 that precedes its anchor. A single layer can submit to several streams;
an event geometrically underneath a label is not sufficient proof of ownership.
Use call sites and CPU/GPU correlation flows for exact attribution. CUDA Graph
replay can lack per-op CPU launches; a separately captured warmup trace may
help identify call sites, but its timings cannot replace steady-state timings.

Do not:

- Count every Top-K launch as a layer; one logical selection can launch several
  kernels, and some layers do no index selection at all.
- Count one entire repeated target/draft cluster as a layer.
- Interpret CPU `record_function` duration as GPU execution duration.
- Stretch the last layer to the next iteration's first anchor.
- Shift or crop original timestamps merely to make an image easier to label.

## Time units and import checks

Chrome JSON `ts` and `dur` are in **microseconds**, regardless of the optional
`displayTimeUnit` presentation setting. Perfetto SQL `ts` and `dur` are in
**nanoseconds**. The report's relative milliseconds subtract the minimum
finite original event timestamp. Compare this origin with `trace_bounds`
before assuming it is the exact UI zero.

Very large floating-point timestamps can introduce 1 ns overlaps when a guide
duration is subtracted in Python and converted to ns by Perfetto. The helper
rounds auxiliary endpoints to integer microseconds (at most 0.5 us adjustment),
keeping the guide on one flat track. Original events are untouched; the report
includes both anchor timestamps and the rounded guide endpoints.

Useful SQL after opening the annotated trace (also works with the official
Trace Processor Python API):

```sql
SELECT name, count(*) AS occurrences
FROM slice WHERE category = 'layer_guide'
GROUP BY name ORDER BY CAST(substr(name, 2) AS INT);

SELECT s.name, s.ts, s.dur, t.name AS track
FROM slice s JOIN track t ON t.id = s.track_id
WHERE s.category = 'layer_guide' ORDER BY s.ts;
```

Expect `num_layers * passes` slices; each layer appears `passes` times. Zero
last-layer duration is intentional without a terminal anchor. Compare original
and annotated JSON event arrays and top-level metadata to verify preservation.
For a CPU/GPU trace, check CPU event categories and flow events too. This helper
loads the JSON in memory; for multi-GB traces, use an existing Trace Processor
and a SQL debug track rather than exhausting local memory with a rewrite.

## Browser viewing and screenshots

The auxiliary track is stored in the trace itself. Manual **Open trace file**
is the simplest path. Pin the guide next to the selected GPU streams, then zoom
into one pass until labels are readable. Keep stream labels and the time axis
visible. Capture one overview and a tighter view when the requested layers
cannot fit legibly together. Capture the real viewer, not a recreated timeline.

If using `serve_local_trace.py`, open its loopback URL in the requested browser.
The page loads the public Perfetto UI and sends the bytes from its single local
`/trace` endpoint via `postMessage`. A PING/PONG handshake prevents lost messages
while the UI initializes. A “Local trace sent” status confirms transfer only;
verify the actual timeline has loaded. Keep the server running while viewing.
No OS accessibility permission changes or browser profile edits are needed.

For direct browser automation, use available authorized browser tools or an
isolated automation profile. Respect the user's choice of browser for the
final view. If the UI cannot be inspected, report that limitation instead of
claiming a screenshot was visually verified.

Official references:

- [Chrome JSON trace import](https://perfetto.dev/docs/getting-started/other-formats)
- [Embedding the Perfetto UI](https://perfetto.dev/docs/visualization/embedding-the-ui)
- [Embedding API reference](https://perfetto.dev/docs/visualization/embedding-api-reference)
- [Trace Processor Python API](https://perfetto.dev/docs/analysis/trace-processor-python)
