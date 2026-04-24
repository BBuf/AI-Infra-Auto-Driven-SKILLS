# SGLang Moss-VL Support and Optimization Timeline

Scope: Moss-VL native SGLang runtime, image/video processor, conversation template, multimodal scheduler metadata, cross-attention custom masks, and flashinfer prefill requirement.

Evidence snapshot: SGLang `origin/main` `bca3dd958` (`2026-04-24`). Full dossier: `skills/model-optimization/sglang/sglang-moss-vl-optimization/references/pr-history.md`.

## Diff-Reviewed PR

#23454 added Moss-VL runtime support. The full diff was reviewed (`3397` lines, `10` files). The PR adds `moss_vl.py`, `multimodal/processors/moss_vl.py`, Moss-VL fields in `schedule_batch.py`, a `moss-vl` conversation template, model-config registration, and a `flashinfer` prefill requirement.

Key contract: Moss-VL vision tokens include separator tokens, frame visibility is converted into a packed cross-attention custom mask, and encoder-prefix placeholder tokens are stripped before text extend.
