# Model Optimization Skills and Standards

This directory stores shared model-optimization workflows rather than
per-model skills. Model-specific optimization records live as PR-driven
documentation under
[`model-pr-optimization-history/`](../../model-pr-optimization-history/).

- `model-pr-diff-dossier/`: shared production standard for manual,
  diff-backed model PR histories across SGLang, vLLM, and future serving
  frameworks.
- `sglang-model-day0-support/`: evidence-driven workflow for architecture gap
  maps, parallel/kernel adaptation, public PR DAGs, seven release gates, and
  sanitized Day-0 support bundles.

Every model PR history should follow the same rule: read the source diff for
each PR and document motivation, implementation, key code excerpt, and
validation/risk.

Open PR watchpoints belong in the relevant framework/model history files, not
in a standalone radar file. Before proposing a new optimization, check the
framework README and per-model history docs for open watchpoints that may not
yet appear in merged git-traced histories.
