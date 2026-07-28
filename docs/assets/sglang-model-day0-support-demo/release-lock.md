# Release Lock

## Source Revisions

| Component | Public source | Immutable revision |
| --- | --- | --- |
| Fictional model | `https://huggingface.co/example/Aurora-Hybrid-70B-VL` | `1111111111111111111111111111111111111111` |
| Fictional processor | `https://huggingface.co/example/Aurora-Hybrid-70B-VL` | `1111111111111111111111111111111111111111` |
| SGLang | `https://github.com/sgl-project/sglang` | `2222222222222222222222222222222222222222` |
| Transformers | `https://github.com/huggingface/transformers` | `3333333333333333333333333333333333333333` |

## Artifacts

| Artifact | Public identifier | Verified relation |
| --- | --- | --- |
| Image | `ghcr.io/example/aurora-day0:2026-08-15` | Build labels record the pinned SGLang revision |
| Weights | `example/Aurora-Hybrid-70B-VL` | Revision is pinned above |
| Cookbook | `docs/models/aurora-hybrid-70b-vl.md` | Commands use the same image and revisions |

## Limitations

- Only NVIDIA B200 TP8 unified and TP4+TP4 PD are claimed.
- Only BF16, one RGB image, and MTP draft lengths up to four are claimed.
- Optimized recurrent and MoE kernels remain optional.
- Any revision change reopens source, load, quality, and release gates.
