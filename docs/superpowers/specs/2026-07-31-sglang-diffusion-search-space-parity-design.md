# SGLang Diffusion Search-Space Parity Design

## Goal

Make the serial SGLang Diffusion optimization flow see substantially the same
optimization opportunities as the reviewed Sol-Engine revision while retaining
SGLang-native implementation, deterministic evidence, and one interactive root
agent.

The flow must also expose stronger implementation knowledge from the locked
SGLang checkout, KDA-Pilot diffusion kernels, KernelWiki, NCU guidance,
FastVideo, and the skills stored in those repositories.

## User Contract

One campaign continues to have exactly one AI owner: the current conversation.
The deterministic Controller may lock and index sources, build a search-space
catalog, route method families, create one worktree, and verify evidence. It
must not spawn an AI process.

Search-space parity has four explicit levels:

1. `documented`: a method family or direction exists in a locked source;
2. `referenced`: a structured candidate manifest or implementation reference
   exists;
3. `adapted`: the method has a real SGLang/model/hardware implementation;
4. `validated`: the adapted method passed the frozen end-to-end gate.

The Controller and root agent must never collapse those levels into a claim
that a documented Sol method is immediately executable in SGLang.

## Approaches Considered

### 1. Full source mirror plus a normalized catalog

Snapshot the complete relevant Sol search surface, derive a deterministic
catalog from its search-space documents and candidate manifests, and bind the
catalog and all knowledge snapshots into every work order. Adapt selected
methods to the locked SGLang source.

This is the chosen approach. It preserves Sol's exploration breadth without
making the SGLang runtime depend on Sol's model-specific implementation.

### 2. Add more prose to the six local technique files

This is small, but method additions remain unstructured and can silently drift
from Sol. The root agent cannot enumerate which candidate manifests or
capabilities were actually considered.

### 3. Import and execute Sol's technique runtime directly

This maximizes code reuse but couples SGLang campaigns to Cosmos3-oriented
adapters, environment variables, and model seams. A successful Sol dry run
would not prove SGLang engagement. This approach is rejected.

## Source And Knowledge Architecture

Every campaign locks these primary repositories:

- SGLang;
- Sol-Engine;
- FastVideo; and
- KDA-Pilot.

KDA-Pilot stores KernelWiki, the NCU report skill, and the warp-specialization
skill as Git submodules. The Controller must read their exact gitlink commits
from the locked KDA-Pilot revision, normalize public GitHub SSH URLs to HTTPS,
and create independent source locks and worktrees for all three. Treating the
parent commit as provenance for unmaterialized submodule files is forbidden.

The allowlisted knowledge snapshot must include:

- Sol `search_space/**`, `candidates/**`, `techniques/**`,
  `site_docs/techniques/**`, model profiles, and workflow scopes;
- SGLang Diffusion skills, benchmark/profile guidance, runtime model and
  pipeline seams, attention/cache/quantization/distributed code, native kernel
  registration, tests, and root kernel-development skills;
- KDA-Pilot diffusion rules plus text source, benchmark, result, export, and
  correctness material for diffusion kernels;
- KernelWiki's skill, indices, queries, references, candidate ledgers, source
  evidence, and query helpers;
- the NCU and warp-specialization skills with their text references and tools;
  and
- FastVideo attention, quantization, kernel, profiling, and optimization
  material.

Snapshots remain text-only, non-executable, path-safe, redacted, commit-bound,
and individually hashed.

## Normalized Sol Search-Space Catalog

During deterministic setup, build `SEARCH-SPACE.json` from the locked Sol
checkout.

The catalog contains exactly six top-level families:

- `kernel`;
- `cache`;
- `sparse_attention`;
- `quantization`;
- `token_pruning`; and
- `topology`.

For each family record:

- the canonical search-space document and its SHA-256;
- every method-family heading or bullet under its method-family section;
- relevant site documentation;
- every structured candidate manifest;
- candidate requirements and capabilities;
- candidate kind, purpose, model profile, implementation reference,
  efficiency primitive, and quality gate; and
- registered runtime techniques and build/load transforms.

Record model-specific top-level candidate files separately as recipes. Preserve
their source paths and metadata without treating their environment or machine
paths as portable SGLang settings.

Fail closed if a canonical search-space document is missing, a structured
candidate has an unknown dimension, a referenced generic implementation is
missing, or an expected family cannot be represented.

## Routing And Technique Scopes

Routes are family-level suggestions; catalog entries are candidate-level
opportunities.

- Always expose `kernel`.
- Expose `topology` when the frozen workload uses more than one GPU.
- When quality-gated optimization is permitted, expose `cache`,
  `sparse_attention`, `quantization`, and `token_pruning`.
- Keep the full catalog visible even when one family is not routed, so a
  single-GPU run still records that topology exists but is inapplicable to the
  frozen resource envelope.

Replace the PISA-only lane with a `sparse_attention` umbrella. PISA remains one
candidate family alongside Sparse VideoGen-style routing, semantic
permutation, AdaSpa-style search/reuse, SpargeAttn-style proxy masks, LVSA,
SVOO, HASTE, MInference-style patterns, and Sol-Attn where supported.

Remove the three-family Cache restriction. Preserve TeaCache, EasyCache, and
Taylor-style forecasting, and also consider whole-step reuse, PAB/attention
broadcast, block/layer and FORA caching, token-wise caching, CFG-aware reuse,
content/motion-adaptive schedules, and architecture-aware reuse.

Kernel remains lossless. Approximate attention and reduced-precision backends
must be evaluated in quality-gated sparse-attention or quantization lanes
rather than weakening the kernel gate.

## Work-Order Knowledge Binding

`AGENT-WORK.json` must include absolute paths and SHA-256 bindings for:

- `KNOWLEDGE.json`;
- `SEARCH-SPACE.json`;
- the frozen baseline and profile;
- the selected technique scope; and
- all source locks.

The root agent must begin a claimed lane by reading the family projection in
the catalog and then query the relevant knowledge snapshots. The precedence is:

1. frozen workload and measured profile;
2. live locked SGLang source and SGLang skills;
3. Sol catalog for opportunity completeness;
4. KDA-Pilot/KernelWiki/NCU material for implementation and profiling;
5. FastVideo and other references for transferable ideas.

Sol references may expand a hypothesis but may not override SGLang's live model
layout or the frozen correctness gate.

Every submitted implementation manifest must cite at least one
`knowledge_origin` entry with source, commit, relative path, and raw source
SHA-256. The verifier resolves each citation through `KNOWLEDGE.json` and
rejects missing, stale, or fabricated provenance.

## Error Handling

- Missing required knowledge files produce setup failure before the baseline.
- KDA submodule drift or an inaccessible exact gitlink commit produces an
  explicit source-lock failure.
- Empty required source snapshots fail instead of silently creating an empty
  knowledge directory.
- A documented or referenced method that lacks model or hardware support is
  recorded as unsupported; it is not counted as a scientific round.
- Candidate manifest and live SGLang capability mismatch is a preflight result,
  not an end-to-end failure.
- Search-space catalog drift is detected through its SHA-256 binding in the
  work order.

## Validation

Tests must prove:

- the knowledge registry contains Sol and the three KDA-derived skill sources;
- hidden SGLang skills and expanded KDA kernel materials are allowlisted;
- KDA submodule commits are derived from a locked parent revision;
- the Sol catalog discovers all six families, structured candidates, recipes,
  method-family entries, and registered implementations;
- malformed or incomplete catalog inputs fail closed;
- routes use `sparse_attention` instead of the PISA-only lane;
- work orders bind the catalog and knowledge manifest;
- implementation manifests require real knowledge provenance;
- the verifier rejects unknown source/path/commit/hash citations; and
- the complete existing unit and integration suite remains green.

No validation may use a subagent or a nested AI process.
