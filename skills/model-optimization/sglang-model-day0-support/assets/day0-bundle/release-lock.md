# Release Lock

## Source Revisions

| Component | Public source | Immutable revision |
| --- | --- | --- |
| Model | `{{MODEL_SOURCE}}` | `{{MODEL_REVISION}}` |
| Tokenizer/processor | `{{PROCESSOR_SOURCE}}` | `{{PROCESSOR_REVISION}}` |
| SGLang | `https://github.com/sgl-project/sglang` | `{{SGLANG_SHA}}` |
| Dependency | `{{DEPENDENCY_SOURCE}}` | `{{DEPENDENCY_REVISION}}` |

## Artifacts

| Artifact | Public identifier | Build/source relation | Verified |
| --- | --- | --- | --- |
| Image | `{{PUBLIC_IMAGE}}` | Built from `{{SGLANG_SHA}}` | `{{IMAGE_STATUS}}` |
| Weights | `{{PUBLIC_WEIGHT_ID}}` | Revision `{{MODEL_REVISION}}` | `{{WEIGHT_STATUS}}` |
| Cookbook | `{{PUBLIC_COOKBOOK_URL}}` | Commands target `{{SGLANG_SHA}}` | `{{COOKBOOK_STATUS}}` |

## Limitations

- Supported hardware: `{{SUPPORTED_HARDWARE}}`
- Supported topology: `{{SUPPORTED_TOPOLOGY}}`
- Unsupported combination: `{{UNSUPPORTED_COMBINATION}}`
- Re-audit trigger: `{{REAUDIT_TRIGGER}}`
