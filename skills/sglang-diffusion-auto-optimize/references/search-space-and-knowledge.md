# Search Space And Knowledge Protocol

## Bound campaign artifacts

Every valid campaign creates:

- `SEARCH-SPACE.json`: the normalized catalog derived from the locked
  Sol-Engine revision;
- `KNOWLEDGE.json`: source-to-snapshot index paths; and
- `knowledge/<source>/<commit>/index.json`: per-file path, raw SHA-256,
  redacted-reference SHA-256, headings, and symbols.

The catalog has six families: kernel, cache, sparse attention, quantization,
token pruning, and topology. Its candidates and recipes are references until
adapted to the locked SGLang source.

## Knowledge-first hypothesis selection

For a claimed family:

1. read its methods, candidates, required capabilities, implementation
   references, and `review_items` in `SEARCH-SPACE.json`;
2. inspect the measured profile and the live SGLang model/pipeline seam;
3. search the SGLang snapshot for the model, hotspot symbols, existing backend
   hooks, and relevant SGLang skills;
4. search Sol for complete opportunity coverage and composition constraints;
5. search KDA-Pilot, KernelWiki, and NCU material for matching shapes,
   implementations, profiling methods, and hardware-specific limitations;
6. search FastVideo for transferable diffusion implementations; and
7. classify each serious direction as documented, referenced, adapted, or
   validated.

Do not load every copied reference into context. Search snapshot indices first,
then read only relevant reference files. Treat scripts in snapshots as
non-executable knowledge; run code only from a reviewed checkout or the
candidate worktree.

## Capability and readiness rules

- `documented` means only that a method appears in a locked search document.
- `referenced` means a candidate manifest, recipe, policy, or implementation
  reference exists.
- `adapted` requires a real SGLang/model/hardware path with an OFF guard.
- `validated` requires positive engagement and the complete frozen end-to-end
  gate.

Reject NVFP4 on hardware without native NVFP4 support. Reject model-specific
token, attention, cache, or topology assumptions when the locked SGLang model
does not expose the required seam. Record those as preflight evidence without
consuming a scientific round.

Before `no_gain`, cover every applicable family item required by the technique
scope. A Sol `env_only` or `blocker_probe` manifest is not an implementation
success.

## Candidate provenance

Every implementation manifest contains at least one exact origin:

```json
{
  "source": "sglang",
  "commit": "40-lowercase-hex",
  "path": "python/sglang/relative/source.py",
  "sha256": "64-lowercase-hex"
}
```

Copy all four values from a bound knowledge index. Cite the most direct sources
that informed the hypothesis: normally the live SGLang seam plus the Sol/KDA
reference used for adaptation. The deterministic verifier rejects unknown
sources, commits, paths, and hashes.
