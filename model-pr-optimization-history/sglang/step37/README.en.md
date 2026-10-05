# SGLang Step 3.7 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/cookbook/autoregressive/StepFun/Step-3.7-Flash.mdx` | no direct PR-number commit |
| `python/sglang/srt/configs/step3p7.py` | [#26565](https://github.com/sgl-project/sglang/pull/26565) |
| `python/sglang/srt/models/step3p7.py` | [#26565](https://github.com/sgl-project/sglang/pull/26565) |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 1
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-29 | [#26565](https://github.com/sgl-project/sglang/pull/26565) | merged | model: Step-3.7-Flash Support | `python/sglang/srt/models/step3p7.py`, `python/sglang/srt/configs/step3p7.py` |

## Per-PR Diff Audit Cards

### PR #26565 - model: Step-3.7-Flash Support

- Link: https://github.com/sgl-project/sglang/pull/26565
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/step3p7.py`, `python/sglang/srt/models/step3p7.py`; associated commits `3bdea78ad11d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +1094/-7, 1284 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output, touching `Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__`; `python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config, touching `Step3p7VisionEncoderConfig, __init__, Step3p7Config`.
- Code diff details:
  - `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output
  - `python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/step3p7.py
@@ -0,0 +1,200 @@
+from typing import Iterable, List, Optional, Tuple
+import torch
+from torch import nn
+from transformers.activations import ACT2FN
+from sglang.srt.configs.step3p7 import Step3p7Config
+from sglang.srt.layers.linear import ColumnParallelLinear
diff -- python/sglang/srt/configs/step3p7.py
@@ -0,0 +1,97 @@
+from typing import Optional, Union
+from transformers.configuration_utils import PretrainedConfig
+class Step3p7VisionEncoderConfig(PretrainedConfig):
+    model_type = "perception_encoder"
+    def __init__(
+        self,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/step3p7.py` added +200/-0; `python/sglang/srt/configs/step3p7.py` added +97/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/step3p5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
