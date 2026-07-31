# Work-Order And Review Protocol

## State And Commands

```text
NEW -> BASELINE_LOCKED -> PROFILED -> AWAITING_AGENT
AWAITING_AGENT -> SEARCHING -> INTEGRATING -> FINAL_VERIFYING
FINAL_VERIFYING -> AWAITING_AGENT | terminal
```

Use:

```bash
sgl-diffusion-engine work --campaign <campaign> --json
sgl-diffusion-engine claim --campaign <campaign> --technique <name>
sgl-diffusion-engine submit --campaign <campaign> \
  --delivery <worktree>/DELIVERY.json
sgl-diffusion-engine skip --campaign <campaign> --technique <name> \
  --classification <unsupported|no_gain|blocked> --reason <reason>
```

`claim` returns absolute paths for `worktree`, `delivery_path`, `review_path`,
the frozen baseline/profile, the technique scope, `KNOWLEDGE.json`, and
`SEARCH-SPACE.json`. It also returns SHA-256 bindings and remaining scientific
rounds.

Before editing, verify those hashes, read the selected family projection, and
query only relevant files from the bound knowledge snapshot indices. Follow
[search-space-and-knowledge.md](search-space-and-knowledge.md).

## Same-Agent Review

Commit the candidate before review. Write `AGENT-REVIEW.json` at the exact
`review_path`:

```json
{
  "schema_version": 1,
  "producer": "interactive-root-agent",
  "campaign_id": "campaign-id",
  "epoch": 1,
  "technique": "kernel",
  "baseline_commit": "40-hex",
  "candidate_commit": "40-hex",
  "diff_sha256": "64-hex",
  "method_argument_sha256": "64-hex",
  "method_equivalent": true,
  "accepted": true,
  "visual_review": {
    "required": false,
    "accepted": true,
    "prompt_count": 0,
    "artifact_sha256": ["64-hex"]
  },
  "findings": []
}
```

Compute `diff_sha256` over the exact UTF-8 output of:

```bash
git diff --binary --full-index <baseline-commit> <candidate-commit> --
```

For a lossless technique, `method_argument_sha256` binds the exact
`method_argument` string in `equivalence.json`. Bind the authenticity verdict
digest in `visual_review.artifact_sha256`.

For a quality-gated technique, bind this canonical no-whitespace JSON string:

```json
{"activation":<implementation-manifest activation>,"technique":"<technique>"}
```

Set `visual_review.required` to `true`, `prompt_count` to `5`, and bind the
`visual_verdict.json` digest. That verdict must contain:

```json
{
  "candidate_id": "candidate-id",
  "overall": "pass",
  "producer": "interactive-root-agent",
  "external_api": false,
  "prompt_evidence": [
    {"prompt": 0, "verdict": "pass"},
    {"prompt": 1, "verdict": "pass"},
    {"prompt": 2, "verdict": "pass"},
    {"prompt": 3, "verdict": "pass"},
    {"prompt": 4, "verdict": "pass"}
  ]
}
```

Do not call this independent review. The same root agent implemented and
reviewed the candidate; deterministic hashing makes that fact auditable.

Every durable `implementation-manifest.json` includes a nonempty
`knowledge_origin`:

```json
[
  {
    "source": "sglang",
    "commit": "40-lowercase-hex",
    "path": "python/sglang/relative/source.py",
    "sha256": "64-lowercase-hex"
  }
]
```

Use exact values from a snapshot index. The verifier resolves every citation
through the bound `KNOWLEDGE.json`.

## Submission Integrity

`submit` records the delivery digest as the scientific-round boundary. Do not
edit the delivery afterward. Verification fails closed on stale commits,
hashes, commands, source files, visual artifacts, or review fields.

After a composed-stack regression, `skip --classification no_gain` may close
one already verified technique. The controller keeps its evidence, removes it
from the selected integration set, advances a selection-only epoch, and
remeasures the remaining verified subset. That operation consumes no
scientific round.
