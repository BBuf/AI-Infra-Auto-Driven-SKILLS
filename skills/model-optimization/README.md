# Model Optimization Skills

This directory stores model-family optimization skills by serving framework.

- `model-pr-diff-dossier/`: shared production standard for manual, diff-backed model PR histories across SGLang, vLLM, and future serving frameworks.
- `sglang/`: SGLang model optimization skills.
- `vllm/`: vLLM model optimization skills.

Every model optimization skill that cites PR history must follow the same rule: read the source diff for each PR and document motivation, implementation, key code excerpt, and validation/risk.

Open PR watchpoints belong in the relevant framework/model history files, not
in a standalone radar file. Before proposing a new optimization, check the
framework README and the model skill's `references/pr-history.md` for open
watchpoints that may not yet appear in merged git-traced histories.
