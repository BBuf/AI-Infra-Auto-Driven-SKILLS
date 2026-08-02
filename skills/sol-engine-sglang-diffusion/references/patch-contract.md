# SGLang Patch Delivery Contract

## Authoritative Candidate

Export only the complete SGLang source tree used by the exact integrated run
accepted in upstream Sol's `INTEGRATED-DELIVERY.json`. Bind it using the run id,
candidate id, frozen base commit, ordered activation/patch digests, benchmark
artifact, and source-tree path recorded by the model adapter.

Do not merge component patches after measurement or choose a different
frontier point during packaging. Any source change after the accepted run
requires another upstream Sol integration measurement and gate.

## Export

Run:

```bash
python <skill-dir>/scripts/extract_sglang_patch.py \
  --base-repo <clean-repository-containing-the-frozen-commit> \
  --base-commit <full-frozen-main-commit> \
  --candidate-tree <complete-accepted-source-tree-without-.git> \
  --output <delivery-dir>/sglang.patch
```

The exporter:

- resolves the base commit through Git;
- requires SGLang's multimodal-generation source sentinel in base and
  candidate;
- rejects candidate Git metadata;
- materializes the candidate in a temporary detached base worktree;
- emits a binary, full-index diff including additions and deletions;
- apply-checks the patch in a second clean detached worktree;
- prints the resolved base and patch SHA-256 digest.

The temporary worktrees are campaign/tool scratch state and are removed after
validation. The base repository's current branch and dirty files are not
modified.

## Delivery Manifest

Write a small `DELIVERY-MANIFEST.json` beside the patch with:

- SGLang origin and frozen full main commit;
- upstream Sol origin, branch, and frozen full commit;
- KDA-Pilot and submodule commits;
- knowledge-manifest path and SHA-256;
- upstream integrated-delivery path and digest;
- accepted integrated point/candidate/run identifiers;
- accepted benchmark and quality/correctness evidence paths;
- accepted run-owned SGLang source path and source-provenance digest;
- patch path and SHA-256;
- exact clean-worktree apply-check command and result.

This manifest reports provenance only. Do not add a second score, quality
judgment, or termination explanation that competes with Sol.

## Content Audit

Inspect patch names and contents before handoff. Reject a patch containing:

- `orchestration/`, `workflow/`, Sol model overlay, or Sol prompts;
- KDA-Pilot, KernelWiki, NCU, history, or knowledge-pack files;
- `runs/`, `output/`, profiler traces, rendered media, checkpoints, caches, or
  build products;
- `.git`, credentials, tokens, private keys, hostnames/IPs that are not real
  product configuration, or absolute machine-local paths;
- unrelated changes already present in a user's worktree.

Apply the patch to a fresh detached worktree at the frozen SGLang commit and
run the relevant SGLang tests there when the target machine remains available.
The tests supplement, but never replace, upstream Sol's accepted performance
and quality evidence.
