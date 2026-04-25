#!/usr/bin/env python3
"""Resolve or generate model architecture diagrams.

The script is intentionally dependency-free so the skill works in a fresh
checkout. Existing diagrams are returned as raw GitHub image URLs. Generated
fallback diagrams are written as local SVG plus Mermaid text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import unicodedata
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = SKILL_DIR / "references" / "diagram-index.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/model-architecture-diagrams")
DEFAULT_COOKBOOK_ROOT = Path("/Users/bbuf/工作目录/Common/sgl-cookbook")
LOCAL_SOURCE_ROOTS = {
    "InfraTech": Path("/tmp/InfraTech"),
    "self-llm": Path("/tmp/self-llm"),
}
MODEL_DOC_DIRS = ("autoregressive", "diffusion", "omni")
MODEL_PATH_RE = re.compile(
    r"--model-path\s+([A-Za-z0-9_.:/-]+)"
    r"|model=['\"]([A-Za-z0-9_.:/-]+)['\"]"
    r"|model_path:\s*([A-Za-z0-9_.:/-]+)"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("_", "-").replace("/", " ")
    text = re.sub(r"[^a-z0-9.\-\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify(text: str) -> str:
    text = normalize(text)
    text = re.sub(r"[^a-z0-9.\-]+", "-", text).strip("-")
    return text or hashlib.sha1(text.encode()).hexdigest()[:10]


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalize(text))


def load_index() -> dict[str, Any]:
    with INDEX_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def alias_score(query: str, aliases: list[str], title: str) -> int:
    q = normalize(query)
    qc = compact(query)
    candidates = [normalize(title), *(normalize(a) for a in aliases)]
    best = 0
    for c in candidates:
        if not c:
            continue
        cc = compact(c)
        if q == c:
            best = max(best, 100)
        elif qc and qc == cc:
            best = max(best, 95)
        elif safe_contains(q, c) or safe_contains(c, q):
            best = max(best, 80)
        else:
            q_tokens = set(q.replace("-", " ").split())
            c_tokens = set(c.replace("-", " ").split())
            if q_tokens and c_tokens:
                overlap = len(q_tokens & c_tokens)
                if overlap:
                    best = max(best, 20 + overlap * 10)
    return best


def safe_contains(needle: str, haystack: str) -> bool:
    """Substring match that avoids version-prefix false positives.

    `deepseek-v3` should not match `deepseek-v3.2`, while `deepseek mla`
    should match `deepseek mla mha`.
    """

    pos = haystack.find(needle)
    if pos < 0:
        return False
    before = haystack[pos - 1] if pos > 0 else " "
    after_pos = pos + len(needle)
    after = haystack[after_pos] if after_pos < len(haystack) else " "
    if before.isalnum():
        return False
    if after.isalnum():
        return False
    if after in ".- " and after_pos + 1 < len(haystack) and needle[-1].isdigit():
        suffix = haystack[after_pos + 1 :]
        suffix_token = suffix.replace("-", " ").split(" ", 1)[0]
        if suffix_token not in {
            "architecture",
            "diagram",
            "model",
            "fp8",
            "bf16",
            "nvfp4",
            "awq",
            "gptq",
            "instruct",
            "thinking",
            "pt",
            "chat",
            "base",
        }:
            return False
    return True


@dataclass(frozen=True)
class Node:
    key: str
    title: str
    subtitle: str
    kind: str


@dataclass(frozen=True)
class Template:
    template_id: str
    description: str
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]
    sglang_files: tuple[str, ...]
    cookbook_hints: tuple[str, ...]


TEMPLATES: dict[str, Template] = {
    "decoder_dense": Template(
        "decoder_dense",
        "Dense decoder-only Transformer/GQA architecture.",
        (
            Node("input", "Text tokens", "token ids + positions", "input"),
            Node("embed", "Embedding", "token embedding + RoPE positions", "embed"),
            Node(
                "block",
                "N x Decoder block",
                "RMSNorm -> Attention -> MLP/SwiGLU",
                "block",
            ),
            Node("attn", "Attention", "MHA/GQA/FlashAttention backend", "attention"),
            Node("mlp", "Dense FFN", "gate/up/down projections", "ffn"),
            Node("out", "Norm + LM head", "logits / next token", "output"),
        ),
        (
            ("input", "embed"),
            ("embed", "block"),
            ("block", "attn"),
            ("attn", "mlp"),
            ("mlp", "out"),
        ),
        ("llama.py", "qwen2.py", "mistral.py", "gemma.py"),
        ("docs/autoregressive",),
    ),
    "decoder_moe": Template(
        "decoder_moe",
        "Decoder-only Transformer with sparse MoE feed-forward layers.",
        (
            Node("input", "Text tokens", "token ids + RoPE positions", "input"),
            Node("embed", "Embedding", "shared token table", "embed"),
            Node(
                "block", "N x Decoder block", "Attention + sparse expert layer", "block"
            ),
            Node("attn", "Attention", "GQA/MQA/FlashAttention", "attention"),
            Node(
                "router", "Router / gate", "top-k expert selection per token", "router"
            ),
            Node("experts", "Experts", "shared expert + routed experts", "moe"),
            Node("out", "Norm + LM head", "logits / next token", "output"),
        ),
        (
            ("input", "embed"),
            ("embed", "block"),
            ("block", "attn"),
            ("attn", "router"),
            ("router", "experts"),
            ("experts", "out"),
        ),
        ("mixtral.py", "gpt_oss.py", "hunyuan.py", "glm4_moe.py", "minimax_m2.py"),
        ("docs/autoregressive",),
    ),
    "swa_moe_mtp": Template(
        "swa_moe_mtp",
        "Sliding-window/global-attention MoE decoder with MTP acceleration.",
        (
            Node("input", "Text tokens", "long-context prompt tokens", "input"),
            Node("embed", "Embedding", "token embedding + RoPE", "embed"),
            Node(
                "pattern",
                "SWA / global attention cycle",
                "mostly sliding-window blocks with periodic full attention",
                "attention",
            ),
            Node(
                "sink", "Attention sink bias", "stabilizes short-window recall", "state"
            ),
            Node("moe", "MoE layer", "top-k routed experts", "moe"),
            Node("mtp", "MTP head", "multi-token prediction draft path", "projector"),
            Node("out", "Norm + LM head", "next token logits", "output"),
        ),
        (
            ("input", "embed"),
            ("embed", "pattern"),
            ("pattern", "sink"),
            ("sink", "moe"),
            ("moe", "mtp"),
            ("mtp", "out"),
        ),
        ("mimo_v2_flash.py", "mimo_mtp.py", "mimo_v2_flash_nextn.py"),
        ("docs/autoregressive/Xiaomi/MiMo-V2-Flash.md",),
    ),
    "mla_moe": Template(
        "mla_moe",
        "MLA attention plus sparse MoE decoder architecture.",
        (
            Node("input", "Text tokens", "token ids + RoPE", "input"),
            Node("embed", "Embedding", "token embedding", "embed"),
            Node("block", "N x MLA-MoE block", "RMSNorm -> MLA -> MoE", "block"),
            Node(
                "mla",
                "MLA attention",
                "latent KV compression + RoPE split",
                "attention",
            ),
            Node("router", "MoE router", "token-level top-k routing", "router"),
            Node("experts", "MoE experts", "routed experts + shared experts", "moe"),
            Node("out", "Norm + LM head", "logits", "output"),
        ),
        (
            ("input", "embed"),
            ("embed", "block"),
            ("block", "mla"),
            ("mla", "router"),
            ("router", "experts"),
            ("experts", "out"),
        ),
        ("deepseek_v2.py", "kimi_k25.py", "glm4_moe.py"),
        (
            "docs/autoregressive/DeepSeek",
            "docs/autoregressive/GLM",
            "docs/autoregressive/Moonshotai",
        ),
    ),
    "dsa_moe": Template(
        "dsa_moe",
        "DSA/NSA sparse-attention decoder with MoE.",
        (
            Node("input", "Text tokens", "long-context token stream", "input"),
            Node("embed", "Embedding", "token embedding + positions", "embed"),
            Node("index", "NSA indexer", "select sparse KV blocks", "attention"),
            Node("block", "N x DSA-MoE block", "DSA/NSA attention + MoE", "block"),
            Node("dsa", "DSA attention", "sparse MLA/NSA backend", "attention"),
            Node("experts", "MoE experts", "routed + shared experts", "moe"),
            Node("out", "Norm + LM head", "logits", "output"),
        ),
        (
            ("input", "embed"),
            ("embed", "index"),
            ("index", "block"),
            ("block", "dsa"),
            ("dsa", "experts"),
            ("experts", "out"),
        ),
        ("deepseek_v2.py", "glm4_moe.py", "nsa_backend.py", "nsa_indexer.py"),
        (
            "docs/autoregressive/DeepSeek/DeepSeek-V3_2.md",
            "docs/autoregressive/GLM/GLM-5.md",
        ),
    ),
    "hybrid_delta_moe": Template(
        "hybrid_delta_moe",
        "Hybrid Gated DeltaNet/Gated Attention plus MoE architecture.",
        (
            Node("input", "Text / multimodal tokens", "long-context sequence", "input"),
            Node("embed", "Embedding", "token + modality embeddings", "embed"),
            Node(
                "cycle",
                "Hybrid block cycle",
                "3 x DeltaNet blocks then 1 x attention block",
                "block",
            ),
            Node(
                "delta",
                "Gated DeltaNet",
                "state-space / linear attention path",
                "state",
            ),
            Node(
                "attn",
                "Gated Attention",
                "periodic full attention refresh",
                "attention",
            ),
            Node("moe", "MoE / FFN", "sparse routed or dense FFN", "moe"),
            Node("out", "Norm + head", "logits or multimodal output", "output"),
        ),
        (
            ("input", "embed"),
            ("embed", "cycle"),
            ("cycle", "delta"),
            ("delta", "attn"),
            ("attn", "moe"),
            ("moe", "out"),
        ),
        ("qwen3_5.py", "qwen3_next.py", "qwen3_vl.py"),
        ("docs/autoregressive/Qwen",),
    ),
    "vlm": Template(
        "vlm",
        "Vision-language model with image/video encoder and LLM decoder.",
        (
            Node("image", "Image / video", "pixels, frames, OCR regions", "input"),
            Node("vision", "Vision encoder", "ViT/SigLIP/MoonViT/DeepStack", "vision"),
            Node(
                "projector", "Projector", "MLP / resampler / pixel shuffle", "projector"
            ),
            Node("text", "Text tokens", "prompt and special tokens", "input"),
            Node("llm", "LLM decoder", "attention + FFN/MoE stack", "block"),
            Node("out", "Output", "text / tool calls / OCR", "output"),
        ),
        (
            ("image", "vision"),
            ("vision", "projector"),
            ("projector", "llm"),
            ("text", "llm"),
            ("llm", "out"),
        ),
        ("qwen3_vl.py", "kimi_vl.py", "internvl.py", "glm4v.py", "ernie45_vl.py"),
        (
            "docs/autoregressive/Qwen/Qwen3-VL.md",
            "docs/autoregressive/GLM",
            "docs/autoregressive/InternVL",
        ),
    ),
    "moonvit_mla_moe": Template(
        "moonvit_mla_moe",
        "Native multimodal MoonViT plus MLA/MoE decoder architecture.",
        (
            Node(
                "image",
                "Image / UI / document",
                "native-resolution visual input",
                "input",
            ),
            Node(
                "moonvit",
                "MoonViT encoder",
                "vision tokens + 2D position features",
                "vision",
            ),
            Node(
                "projector",
                "Projector",
                "pixel shuffle / MLP to LLM space",
                "projector",
            ),
            Node("text", "Text / tool context", "prompt, tools, agent state", "input"),
            Node(
                "decoder",
                "MLA + MoE decoder",
                "long-context sparse expert language stack",
                "block",
            ),
            Node(
                "parsers",
                "Reasoning / tool parsers",
                "thinking + structured tool calls",
                "router",
            ),
            Node(
                "out", "Agentic response", "text, code, actions, tool calls", "output"
            ),
        ),
        (
            ("image", "moonvit"),
            ("moonvit", "projector"),
            ("projector", "decoder"),
            ("text", "decoder"),
            ("decoder", "parsers"),
            ("parsers", "out"),
        ),
        ("kimi_vl.py", "kimi_vl_moonvit.py", "kimi_k25.py", "deepseek_v2.py"),
        (
            "docs/autoregressive/Moonshotai/Kimi-K2.5.md",
            "docs/autoregressive/Moonshotai/Kimi-K2.6.md",
        ),
    ),
    "diffusion_dit": Template(
        "diffusion_dit",
        "Text/image/video diffusion pipeline with DiT denoiser.",
        (
            Node(
                "prompt",
                "Prompt / conditioning",
                "text, image, or video condition",
                "input",
            ),
            Node(
                "encoder", "Text / vision encoder", "conditioning embeddings", "vision"
            ),
            Node("latent", "Latent noise", "scheduler timesteps", "embed"),
            Node(
                "dit", "N x DiT denoiser", "attention + MLP over latent tokens", "block"
            ),
            Node("cache", "Acceleration", "Cache-DiT / TeaCache / CUDA graph", "state"),
            Node("vae", "VAE decode", "latents to pixels/frames", "projector"),
            Node("out", "Generated media", "image or video", "output"),
        ),
        (
            ("prompt", "encoder"),
            ("encoder", "dit"),
            ("latent", "dit"),
            ("dit", "cache"),
            ("cache", "vae"),
            ("vae", "out"),
        ),
        (
            "multimodal_gen/configs/pipeline_configs",
            "multimodal_gen/runtime/models/dits",
        ),
        ("docs/diffusion",),
    ),
    "mova_bimodal_dit": Template(
        "mova_bimodal_dit",
        "Asymmetric video/audio dual-tower diffusion architecture.",
        (
            Node(
                "prompt",
                "Prompt + reference image",
                "video and audio conditions",
                "input",
            ),
            Node("video", "Video tower", "visual latent denoising stream", "vision"),
            Node(
                "audio",
                "Audio tower",
                "synchronized acoustic latent stream",
                "attention",
            ),
            Node(
                "cross",
                "Bidirectional cross-attention",
                "aligns video motion and audio events",
                "block",
            ),
            Node("cache", "Acceleration", "Cache-DiT / USP / torch.compile", "state"),
            Node(
                "decode", "Decoders", "VAE/video frames + audio waveform", "projector"
            ),
            Node("out", "Video + audio", "synchronized generated media", "output"),
        ),
        (
            ("prompt", "video"),
            ("prompt", "audio"),
            ("video", "cross"),
            ("audio", "cross"),
            ("cross", "cache"),
            ("cache", "decode"),
            ("decode", "out"),
        ),
        (
            "multimodal_gen/configs/pipeline_configs",
            "multimodal_gen/runtime/models/dits",
        ),
        ("docs/diffusion/MOVA/MOVA.md",),
    ),
    "diffusion_moe_video": Template(
        "diffusion_moe_video",
        "Video diffusion with timestep-specialized MoE denoisers.",
        (
            Node(
                "prompt",
                "Prompt / image condition",
                "text + optional reference media",
                "input",
            ),
            Node("encoder", "Encoders", "text encoder + VAE/image encoder", "vision"),
            Node("latent", "Video latents", "T x H x W latent tokens", "embed"),
            Node(
                "router", "Timestep router", "routes denoise stage to expert", "router"
            ),
            Node(
                "experts",
                "Denoise experts",
                "specialized DiT experts across timesteps",
                "moe",
            ),
            Node("vae", "VAE decode", "video frames", "projector"),
            Node("out", "Generated video", "frames / mp4", "output"),
        ),
        (
            ("prompt", "encoder"),
            ("encoder", "latent"),
            ("latent", "router"),
            ("router", "experts"),
            ("experts", "vae"),
            ("vae", "out"),
        ),
        (
            "multimodal_gen/configs/pipeline_configs/wan.py",
            "multimodal_gen/runtime/models/dits",
        ),
        ("docs/diffusion/Wan",),
    ),
    "llada": Template(
        "llada",
        "Diffusion language model / masked denoising architecture.",
        (
            Node("input", "Masked tokens", "corrupted sequence", "input"),
            Node("embed", "Embedding", "token + timestep/noise conditioning", "embed"),
            Node(
                "block",
                "N x Denoising block",
                "bidirectional Transformer/MoE stack",
                "block",
            ),
            Node(
                "denoise",
                "Iterative denoising",
                "mask prediction and refinement",
                "state",
            ),
            Node("out", "Decoded tokens", "final text", "output"),
        ),
        (
            ("input", "embed"),
            ("embed", "block"),
            ("block", "denoise"),
            ("denoise", "out"),
        ),
        ("llada2.py",),
        ("docs/autoregressive/LLaDA",),
    ),
    "reranker": Template(
        "reranker",
        "Encoder-style reranker architecture.",
        (
            Node("input", "Query + document", "paired text input", "input"),
            Node(
                "encoder",
                "Transformer encoder",
                "bidirectional attention stack",
                "block",
            ),
            Node("pool", "Pooling / classifier", "score head", "projector"),
            Node("out", "Relevance score", "ranking output", "output"),
        ),
        (("input", "encoder"), ("encoder", "pool"), ("pool", "out")),
        ("bert.py", "roberta.py"),
        ("docs/autoregressive/Jina",),
    ),
    "audio_hybrid_serving": Template(
        "audio_hybrid_serving",
        "Hybrid speech serving stack with an outer audio server and SGLang inner engine.",
        (
            Node("audio_in", "Audio input", "speech prompt / reference audio", "input"),
            Node("server", "FlashLabs server", "audio I/O, state, batching=1", "state"),
            Node(
                "sglang",
                "SGLang inner engine",
                "accelerated supported inference loops",
                "block",
            ),
            Node(
                "state",
                "Custom state",
                "KV cache and audio generation state",
                "attention",
            ),
            Node("audio_out", "Audio output", "speech waveform response", "output"),
        ),
        (
            ("audio_in", "server"),
            ("server", "sglang"),
            ("sglang", "state"),
            ("state", "audio_out"),
        ),
        ("external Chroma-SGLang server",),
        ("docs/autoregressive/FlashLabs/Chroma1.0.md",),
    ),
    "tts_dual_ar": Template(
        "tts_dual_ar",
        "Dual-autoregressive TTS pipeline with RVQ codec and DAC vocoder.",
        (
            Node(
                "text",
                "Text + references",
                "prompt text and optional voice clone audio",
                "input",
            ),
            Node("codec", "RVQ codec", "reference audio to VQ codes", "embed"),
            Node(
                "slow",
                "Slow AR in SGLang",
                "semantic token generation over time",
                "block",
            ),
            Node("fast", "Fast AR head", "9 residual codebooks per step", "attention"),
            Node(
                "cache",
                "Paged KV cache",
                "radix prefix cache + FlashAttention",
                "state",
            ),
            Node("vocoder", "DAC vocoder", "codebook indices to waveform", "projector"),
            Node("out", "Speech audio", "generated / cloned voice", "output"),
        ),
        (
            ("text", "slow"),
            ("codec", "slow"),
            ("slow", "fast"),
            ("fast", "cache"),
            ("cache", "vocoder"),
            ("vocoder", "out"),
        ),
        ("sglang-omni s2pro stack",),
        ("docs/omni/S2-Pro/S2-Pro.md",),
    ),
}


RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("audio_hybrid_serving", ("chroma", "flashlabs")),
    ("tts_dual_ar", ("fish ?audio", "s2[ ._-]?pro", "tts", "dual[ ._-]?ar")),
    (
        "moonvit_mla_moe",
        ("kimi[ ._-]?k2[ ._-]?(5|6)", "kimi[ ._-]?k25", "kimi[ ._-]?k26"),
    ),
    (
        "dsa_moe",
        ("deepseek.*v3[ ._-]?2", "glm[ ._-]?5", "glm[ ._-]?5[ ._-]?1", "dsa", "nsa"),
    ),
    (
        "vlm",
        (
            "vl",
            "vision",
            "ocr",
            "omni",
            "asr",
            "deepseek.*ocr",
            "gemma[ ._-]?4",
            "llama[ ._-]?4",
            "mistral[ ._-]?small[ ._-]?4",
            "step3[ ._-]?vl",
            "internvl",
            "intern[ ._-]?s1",
            "ernie.*vl",
            "glm.*v",
        ),
    ),
    (
        "mla_moe",
        (
            "deepseek",
            "kimi[ ._-]?k2",
            "kimi[ ._-]?k25",
            "kimi[ ._-]?k2[ ._-]?5",
            "hunyuan",
        ),
    ),
    (
        "hybrid_delta_moe",
        (
            "qwen3[ ._-]?next",
            "qwen3[ ._-]?5",
            "qwen3[ ._-]?6",
            "qwen3[ ._-]?coder[ ._-]?next",
            "ring[ ._-]?2",
            "ling[ ._-]?2",
            "kimi[ ._-]?linear",
        ),
    ),
    ("mova_bimodal_dit", ("mova", "video.*audio")),
    ("diffusion_moe_video", ("wan2", "wan-", "video.*diffusion")),
    (
        "diffusion_dit",
        (
            "qwen[ ._-]?image",
            "flux",
            "z[ ._-]?image",
            "mova",
            "chroma",
            "diffusion",
            "dit",
        ),
    ),
    ("swa_moe_mtp", ("mimo", "swa", "sliding.*window.*mtp")),
    ("llada", ("llada", "diffusion language")),
    ("reranker", ("jina.*reranker", "reranker")),
    (
        "decoder_moe",
        (
            "moe",
            "gpt[ ._-]?oss",
            "mixtral",
            "nemotron",
            "ernie",
            "glm",
            "glyph",
            "minimax",
            "qwen3[ ._-]?coder",
            "qwen3",
            "gemma[ ._-]?4.*26b",
            "llama[ ._-]?4",
        ),
    ),
)


def find_existing(query: str, max_results: int) -> list[dict[str, Any]]:
    index = load_index()
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for entry in index["diagrams"]:
        score = alias_score(query, entry.get("aliases", []), entry["title"])
        if score >= 70:
            matches.append((score, -int(entry.get("rank", 50)), entry))
    matches.sort(key=lambda item: (item[0], item[1], item[2]["title"]), reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, entry in matches:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        deduped.append(entry)
        if len(deduped) >= max_results:
            break
    return deduped


def find_existing_any(queries: list[str], max_results: int) -> list[dict[str, Any]]:
    index = load_index()
    scored: dict[str, tuple[int, dict[str, Any]]] = {}
    for query in queries:
        for entry in index["diagrams"]:
            score = alias_score(query, entry.get("aliases", []), entry["title"])
            if score < 70:
                continue
            old = scored.get(entry["id"])
            if old is None or score > old[0]:
                scored[entry["id"]] = (score, entry)
    matches = sorted(
        scored.values(),
        key=lambda item: (
            item[0],
            -int(item[1].get("rank", 50)),
            item[1]["title"],
        ),
        reverse=True,
    )
    return [entry for _, entry in matches[:max_results]]


def infer_template(query: str) -> Template:
    q = normalize(query)
    for template_id, patterns in RULES:
        if any(re.search(pattern, q) for pattern in patterns):
            return TEMPLATES[template_id]
    return TEMPLATES["decoder_dense"]


def mermaid_for(model: str, template: Template) -> str:
    lines = [
        "flowchart TD",
        f'  title["{model} architecture"]',
        "  title:::title",
    ]
    for node in template.nodes:
        label = f"{node.title}<br/><span>{node.subtitle}</span>"
        lines.append(f'  {node.key}["{label}"]:::{node.kind}')
    if template.nodes:
        lines.append(f"  title --> {template.nodes[0].key}")
    for src, dst in template.edges:
        lines.append(f"  {src} --> {dst}")
    lines.extend(
        [
            "  classDef title fill:#111827,color:#ffffff,stroke:#111827,stroke-width:1px;",
            "  classDef input fill:#E0F2FE,stroke:#0284C7,color:#0F172A;",
            "  classDef embed fill:#DCFCE7,stroke:#16A34A,color:#0F172A;",
            "  classDef block fill:#FEF3C7,stroke:#D97706,color:#0F172A;",
            "  classDef attention fill:#EDE9FE,stroke:#7C3AED,color:#0F172A;",
            "  classDef state fill:#FCE7F3,stroke:#DB2777,color:#0F172A;",
            "  classDef router fill:#FFE4E6,stroke:#E11D48,color:#0F172A;",
            "  classDef moe fill:#FFEDD5,stroke:#EA580C,color:#0F172A;",
            "  classDef ffn fill:#F1F5F9,stroke:#475569,color:#0F172A;",
            "  classDef vision fill:#DBEAFE,stroke:#2563EB,color:#0F172A;",
            "  classDef projector fill:#CCFBF1,stroke:#0D9488,color:#0F172A;",
            "  classDef output fill:#DCFCE7,stroke:#15803D,color:#0F172A;",
        ]
    )
    return "\n".join(lines) + "\n"


COLORS = {
    "input": ("#E0F2FE", "#0284C7"),
    "embed": ("#DCFCE7", "#16A34A"),
    "block": ("#FEF3C7", "#D97706"),
    "attention": ("#EDE9FE", "#7C3AED"),
    "state": ("#FCE7F3", "#DB2777"),
    "router": ("#FFE4E6", "#E11D48"),
    "moe": ("#FFEDD5", "#EA580C"),
    "ffn": ("#F1F5F9", "#475569"),
    "vision": ("#DBEAFE", "#2563EB"),
    "projector": ("#CCFBF1", "#0D9488"),
    "output": ("#DCFCE7", "#15803D"),
}


def wrap_svg_text(text: str, width: int = 34) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]


def primary_style(model: str, template: Template) -> tuple[str, str]:
    """Match the public InfraTech palette family for generated diagrams."""

    q = normalize(model)
    if "kimi" in q:
        return ("#5B5B5B", "#1F2937")
    if "minimax" in q or "step" in q:
        return ("#FB7573", "#EF4444")
    if "qwen" in q:
        return ("#7C6EF2", "#5B21B6")
    if "deepseek" in q or "glm" in q:
        return ("#7288F4", "#2563EB")
    if template.template_id.startswith("diffusion") or "mova" in q:
        return ("#F59E0B", "#D97706")
    if template.template_id in {"vlm", "moonvit_mla_moe"}:
        return ("#4CC9C0", "#0D9488")
    return ("#7288F4", "#2563EB")


def _shape(
    out: list[str],
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    *,
    fill: str,
    stroke: str,
    text_fill: str = "#FFFFFF",
    rx: int = 8,
    font_size: int = 22,
    weight: int = 400,
    subtitle: str | None = None,
    dashed: bool = False,
    kind: str = "rect",
) -> None:
    dash = ' stroke-dasharray="14 9"' if dashed else ""
    if kind == "trapezoid":
        inset = min(42, max(16, w // 7))
        points = f"{x + inset},{y} {x + w - inset},{y} " f"{x + w},{y + h} {x},{y + h}"
        out.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2"{dash}/>'
        )
    else:
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
        )
    title_y = y + h / 2 + (font_size * 0.34 if subtitle is None else -4)
    out.append(
        f'<text x="{x + w / 2:.1f}" y="{title_y:.1f}" text-anchor="middle" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="{font_size}" '
        f'font-weight="{weight}" fill="{text_fill}">{escape(title)}</text>'
    )
    if subtitle:
        out.append(
            f'<text x="{x + w / 2:.1f}" y="{y + h - 13}" text-anchor="middle" '
            f'font-family="Arial,Helvetica,sans-serif" font-size="13" '
            f'fill="{text_fill}">{escape(subtitle)}</text>'
        )


def _panel(
    out: list[str],
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    stroke: str,
    *,
    font_size: int = 32,
    title_color: str | None = None,
) -> None:
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" '
        f'stroke="{stroke}" stroke-width="2.3" stroke-dasharray="16 10"/>'
    )
    out.append(
        f'<text x="{x + 18}" y="{y + 40}" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="{font_size}" '
        f'font-weight="700" fill="{title_color or stroke}">{escape(title)}</text>'
    )


def _text(
    out: list[str],
    text: str,
    x: int,
    y: int,
    *,
    size: int = 18,
    fill: str = "#111111",
    weight: int = 400,
    anchor: str = "start",
) -> None:
    out.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(text)}</text>'
    )


def _arrow(
    out: list[str],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    stroke: str = "#111111",
    width: float = 2.2,
    dashed: bool = False,
    marker: str = "arrow",
) -> None:
    dash = ' stroke-dasharray="12 8"' if dashed else ""
    out.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash} '
        f'marker-end="url(#{marker})"/>'
    )


def _polyline(
    out: list[str],
    points: tuple[tuple[int, int], ...],
    *,
    stroke: str = "#111111",
    width: float = 2.2,
    dashed: bool = False,
    marker: str = "arrow",
) -> None:
    dash = ' stroke-dasharray="12 8"' if dashed else ""
    point_text = " ".join(f"{x},{y}" for x, y in points)
    out.append(
        f'<polyline points="{point_text}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}"{dash} marker-end="url(#{marker})"/>'
    )


def _plus(
    out: list[str], cx: int, cy: int, *, fill: str, stroke: str = "#111111"
) -> None:
    out.append(
        f'<circle cx="{cx}" cy="{cy}" r="21" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="2"/>'
    )
    out.append(
        f'<line x1="{cx - 16}" y1="{cy}" x2="{cx + 16}" y2="{cy}" '
        f'stroke="{stroke}" stroke-width="2"/>'
    )
    out.append(
        f'<line x1="{cx}" y1="{cy - 16}" x2="{cx}" y2="{cy + 16}" '
        f'stroke="{stroke}" stroke-width="2"/>'
    )


def _mult(out: list[str], cx: int, cy: int, *, fill: str = "#8DD348") -> None:
    out.append(
        f'<circle cx="{cx}" cy="{cy}" r="24" fill="{fill}" stroke="#111111" '
        f'stroke-width="2"/>'
    )
    out.append(
        f'<line x1="{cx - 15}" y1="{cy - 15}" x2="{cx + 15}" y2="{cy + 15}" '
        f'stroke="#111111" stroke-width="2"/>'
    )
    out.append(
        f'<line x1="{cx - 15}" y1="{cy + 15}" x2="{cx + 15}" y2="{cy - 15}" '
        f'stroke="#111111" stroke-width="2"/>'
    )


def _shape_label(out: list[str], label: str, x: int, y: int) -> None:
    for i, line in enumerate(wrap_svg_text(label, width=30)):
        _text(out, line, x, y + i * 20, size=16)


def _source_note(template: Template) -> str:
    files = ", ".join(template.sglang_files[:4])
    return f"SGLang refs: {files}"


def _layer_label(template: Template) -> tuple[str, str, str]:
    if template.template_id == "decoder_dense":
        return ("N x", "Decoder layer", "Attention + dense FFN")
    if template.template_id in {"mla_moe", "moonvit_mla_moe"}:
        return ("N x", "MLA-MoE layer", "RMSNorm -> MLA -> MoE")
    if template.template_id == "dsa_moe":
        return ("N x", "DSA-MoE layer", "RMSNorm -> DSA/NSA -> MoE")
    if template.template_id == "hybrid_delta_moe":
        return ("3+1 x", "Hybrid cycle", "DeltaNet blocks + gated attention")
    if template.template_id == "swa_moe_mtp":
        return ("N x", "SWA-MoE layer", "Sliding-window/global attention + MoE")
    if template.template_id == "vlm":
        return ("N x", "Vision + LLM", "ViT/projector tokens into decoder")
    if (
        template.template_id.startswith("diffusion")
        or template.template_id == "mova_bimodal_dit"
    ):
        return ("T x", "Denoise step", "conditioned DiT block")
    if template.template_id == "tts_dual_ar":
        return ("2 x", "AR stack", "slow semantic AR + fast codebook AR")
    if template.template_id == "audio_hybrid_serving":
        return ("1 x", "Serving loop", "audio state + SGLang engine")
    if template.template_id == "reranker":
        return ("N x", "Encoder block", "bidirectional attention + score head")
    if template.template_id == "llada":
        return ("T x", "Denoising step", "masked token refinement")
    return ("N x", "MoE layer", "Attention + sparse expert layer")


def _spine_texts(template: Template) -> dict[str, str]:
    defaults = {
        "input": "Text / media",
        "processor": "Tokenizer / processor",
        "embed": "Embedding",
        "shape": "[seq_len, hidden_size]",
        "norm": "RMSNorm",
        "top_norm": "RMSNorm",
        "output": "LM-Head / Output",
    }
    if template.template_id in {"diffusion_dit", "diffusion_moe_video"}:
        defaults.update(
            {
                "input": "Prompt / noise",
                "processor": "Text / image encoder",
                "embed": "Latent tokens",
                "shape": "[frames, h*w, channels]",
                "norm": "AdaLayerNorm",
                "top_norm": "Scheduler step",
                "output": "VAE / media output",
            }
        )
    elif template.template_id == "mova_bimodal_dit":
        defaults.update(
            {
                "input": "Prompt + reference",
                "processor": "Video/audio encoder",
                "embed": "Bimodal latents",
                "shape": "[video/audio latent tokens]",
                "norm": "AdaLayerNorm",
                "top_norm": "Dual decoder",
                "output": "Video + audio output",
            }
        )
    elif template.template_id == "tts_dual_ar":
        defaults.update(
            {
                "input": "Text + reference audio",
                "processor": "Text tokenizer + RVQ",
                "embed": "Semantic/audio codes",
                "shape": "[time, codebook_dim]",
                "norm": "LayerNorm",
                "top_norm": "DAC vocoder",
                "output": "Speech waveform",
            }
        )
    elif template.template_id == "audio_hybrid_serving":
        defaults.update(
            {
                "input": "Audio request",
                "processor": "Audio I/O server",
                "embed": "Audio + KV state",
                "shape": "[stream state]",
                "norm": "State update",
                "top_norm": "SGLang engine",
                "output": "Audio response",
            }
        )
    elif template.template_id == "reranker":
        defaults.update(
            {
                "input": "Query + document",
                "processor": "Pair tokenizer",
                "embed": "Token embeddings",
                "shape": "[pair_len, hidden_size]",
                "norm": "LayerNorm",
                "top_norm": "Pooling",
                "output": "Relevance score",
            }
        )
    elif template.template_id == "llada":
        defaults.update(
            {
                "input": "Masked text",
                "processor": "Noise schedule",
                "embed": "Token + timestep embedding",
                "shape": "[seq_len, hidden_size]",
                "norm": "LayerNorm",
                "top_norm": "Denoise head",
                "output": "Decoded tokens",
            }
        )
    elif template.template_id in {"vlm", "moonvit_mla_moe"}:
        defaults.update(
            {
                "input": "Image/video + text",
                "processor": "Vision processor + tokenizer",
                "embed": "Visual/text embeddings",
            }
        )
    return defaults


def _draw_left_spine(
    out: list[str],
    model: str,
    template: Template,
    primary: tuple[str, str],
) -> dict[str, tuple[int, int]]:
    fill, stroke = primary
    x = 95
    w = 390
    spine = _spine_texts(template)
    label_repeat, layer_title, layer_subtitle = _layer_label(template)
    _text(out, spine["input"], x + 155, 1110, size=20, weight=700)
    _arrow(out, x + w // 2, 1090, x + w // 2, 1048)
    _shape(out, x, 985, w, 58, spine["processor"], fill=fill, stroke=stroke)
    _arrow(out, x + w // 2, 985, x + w // 2, 940)
    _shape(out, x, 875, w, 64, spine["embed"], fill=fill, stroke=stroke)
    _shape_label(out, spine["shape"], x + 280, 858)

    group_x = x - 45
    group_y = 315
    group_w = w + 155
    group_h = 500
    _text(out, label_repeat, max(10, group_x - 58), group_y + 58, size=28, weight=700)
    _panel(
        out, group_x, group_y, group_w, group_h, layer_title, "#111111", font_size=20
    )

    plus_fill = stroke if stroke != "#1F2937" else "#5B5B5B"
    _arrow(out, x + w // 2, 875, x + w // 2, group_y + group_h - 50)
    _plus(out, x + w // 2, group_y + group_h - 54, fill=plus_fill)
    _polyline(
        out,
        (
            (x + 18, group_y + group_h - 38),
            (x + 18, group_y + 270),
            (x + w // 2 - 25, group_y + 270),
        ),
    )
    _shape(
        out,
        x,
        group_y + 370,
        w,
        64,
        spine["norm"],
        fill=fill,
        stroke=stroke,
    )
    _arrow(out, x + w // 2, group_y + 370, x + w // 2, group_y + 327)
    attention_name = {
        "dsa_moe": "DSA",
        "mla_moe": "MLA",
        "moonvit_mla_moe": "MLA",
        "hybrid_delta_moe": "DeltaNet / Attention",
        "swa_moe_mtp": "SWA / Full Attention",
        "vlm": "Vision-LLM Bridge",
        "diffusion_dit": "DiT Attention",
        "diffusion_moe_video": "DiT Expert Router",
        "mova_bimodal_dit": "Cross Attention",
        "tts_dual_ar": "AR Attention",
        "audio_hybrid_serving": "Stateful Engine",
        "reranker": "Self Attention",
        "llada": "Denoising Attention",
    }.get(template.template_id, "Attention")
    _shape(
        out,
        x + 50,
        group_y + 285,
        w - 100,
        58,
        attention_name,
        fill="#FFFFFF",
        stroke="#F59E0B",
        text_fill="#F59E0B",
        dashed=True,
        font_size=24,
        weight=700,
    )
    _arrow(out, x + w // 2, group_y + 285, x + w // 2, group_y + 242)
    _plus(out, x + w // 2, group_y + 240, fill=plus_fill)
    _polyline(
        out,
        (
            (x + 18, group_y + 252),
            (x + 18, group_y + 115),
            (x + w // 2 - 25, group_y + 115),
        ),
    )
    _shape(
        out,
        x,
        group_y + 165,
        w,
        64,
        spine["norm"],
        fill=fill,
        stroke=stroke,
    )
    _arrow(out, x + w // 2, group_y + 165, x + w // 2, group_y + 120)
    ffn_label = "MoE" if any(n.kind == "moe" for n in template.nodes) else "FFN"
    if template.template_id in {"diffusion_dit", "mova_bimodal_dit"}:
        ffn_label = "MLP / Modulation"
    if template.template_id == "tts_dual_ar":
        ffn_label = "Codebook head"
    _shape(
        out,
        x + 50,
        group_y + 82,
        w - 100,
        58,
        ffn_label,
        fill="#FFFFFF",
        stroke="#3B82F6" if ffn_label == "MoE" else "#8DD348",
        text_fill="#3B82F6" if ffn_label == "MoE" else "#8DD348",
        dashed=True,
        font_size=24,
        weight=700,
    )
    _arrow(out, x + w // 2, group_y + 82, x + w // 2, group_y + 38)
    _plus(out, x + w // 2, group_y + 35, fill=plus_fill)

    _arrow(out, x + w // 2, group_y, x + w // 2, 260)
    _shape(out, x, 195, w, 64, spine["top_norm"], fill=fill, stroke=stroke)
    _arrow(out, x + w // 2, 195, x + w // 2, 154)
    _shape(out, x, 90, w, 64, spine["output"], fill=fill, stroke=stroke)
    if template.template_id in {
        "decoder_moe",
        "mla_moe",
        "dsa_moe",
        "swa_moe_mtp",
        "moonvit_mla_moe",
    }:
        _arrow(out, x + w // 2, 90, x + w // 2, 44)
        _shape(
            out,
            x + 75,
            10,
            w - 150,
            58,
            "MTP",
            fill="#FFFFFF",
            stroke="#111111",
            text_fill="#111111",
            dashed=True,
            font_size=22,
            weight=700,
        )

    _text(out, layer_subtitle, group_x + 18, group_y + group_h + 36, size=16)
    return {
        "attention": (x + w, group_y + 315),
        "ffn": (x + w, group_y + 110),
        "input": (x + w, 905),
    }


def _draw_mlp_panel(
    out: list[str], x: int, y: int, w: int, h: int, title: str = "MLP(FFN)"
) -> None:
    green = "#8DD348"
    _panel(out, x, y, w, h, title, green, font_size=30)
    mid = x + 150
    _shape(
        out,
        x + 70,
        y + h - 76,
        220,
        54,
        "Linear(up_proj)",
        fill=green,
        stroke="#169BFF",
    )
    _shape(
        out,
        x + w - 280,
        y + h - 76,
        230,
        54,
        "Linear(gate_proj)",
        fill=green,
        stroke="#169BFF",
    )
    _shape(
        out,
        x + w - 270,
        y + h - 170,
        230,
        54,
        "SiLU Activation",
        fill=green,
        stroke="#169BFF",
    )
    _mult(out, x + w // 2, y + h - 140)
    _shape(
        out,
        mid,
        y + 72,
        270,
        54,
        "Linear(down_proj)",
        fill=green,
        stroke="#169BFF",
    )
    _arrow(out, x + w // 2, y + h - 22, x + w // 2, y + h - 95)
    _arrow(out, x + w - 165, y + h - 76, x + w - 165, y + h - 170)
    _arrow(out, x + w - 165, y + h - 170, x + w // 2 + 25, y + h - 140)
    _arrow(out, x + 180, y + h - 76, x + w // 2 - 24, y + h - 140)
    _arrow(out, x + w // 2, y + h - 116, x + w // 2, y + 126)
    _shape_label(out, "[seq_len, hidden_size]", x + w - 235, y + 57)
    _shape_label(out, "[seq_len, intermediate_size]", x + 45, y + h - 120)


def _draw_moe_panel(
    out: list[str],
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    experts: str = "routed experts",
    router_fill: str = "#2EA8F7",
) -> None:
    blue = "#3B82F6"
    green = "#8DD348"
    _panel(out, x, y, w, h, "MoE", blue, font_size=32)
    router_x = x + w // 2 - 90
    router_y = y + h - 86
    _shape(
        out,
        router_x,
        router_y,
        180,
        56,
        "Router",
        fill=router_fill,
        stroke="#0284C7",
        font_size=20,
    )
    _shape(out, x + 80, y + 120, 180, 54, "MLP", fill=green, stroke="#169BFF")
    _shape(out, x + w // 2 + 20, y + 120, 180, 54, "MLP", fill=green, stroke="#169BFF")
    _shape(out, x + w - 220, y + 120, 160, 54, "MLP", fill=green, stroke="#169BFF")
    _text(out, experts, x + 84, y + 100, size=22, fill=blue, weight=700)
    _text(out, "1 Shared", x + w - 215, y + 100, size=22, fill=blue, weight=700)
    _text(out, "top-k active", x + 70, y + h - 98, size=20, fill="#EF0000", weight=700)
    _text(out, "per token", x + 70, y + h - 70, size=18)
    _text(out, "...", x + w // 2 - 30, y + 157, size=34, weight=700)
    _arrow(out, router_x + 90, router_y, router_x + 90, y + 200)
    for ex, ey in (
        (x + 170, y + 174),
        (x + w // 2 + 110, y + 174),
        (x + w - 140, y + 174),
    ):
        _arrow(out, router_x + 90, router_y, ex, ey, dashed=True)
        _arrow(out, ex, ey, x + w // 2, y + 70, dashed=True)
    _plus(out, x + w // 2, y + 65, fill="#2EA8F7")
    _arrow(out, x + w // 2, y + 32, x + w // 2, y - 26)


def _draw_attention_panel(
    out: list[str], x: int, y: int, w: int, h: int, title: str
) -> None:
    orange = "#FF9F1A"
    red = "#F00000"
    _panel(out, x, y, w, h, title, orange, font_size=30)
    bottom_y = y + h - 82
    _shape(
        out,
        x + 90,
        bottom_y,
        260,
        54,
        "Linear(Q down_proj)",
        fill=orange,
        stroke=orange,
        kind="trapezoid",
    )
    _shape(
        out,
        x + 455,
        bottom_y,
        330,
        54,
        "Linear(KV down_proj)",
        fill=orange,
        stroke=orange,
        kind="trapezoid",
    )
    _shape(out, x + 140, bottom_y - 105, 200, 48, "RMSNorm", fill=orange, stroke=orange)
    _shape(out, x + 520, bottom_y - 105, 200, 48, "RMSNorm", fill=orange, stroke=orange)
    _shape(
        out,
        x + 140,
        bottom_y - 200,
        220,
        50,
        "Linear(Q up_proj)",
        fill=orange,
        stroke=orange,
        kind="trapezoid",
    )
    _shape(
        out,
        x + 500,
        bottom_y - 200,
        260,
        50,
        "Linear(KV up_proj)",
        fill=orange,
        stroke=orange,
        kind="trapezoid",
    )
    _shape(out, x + 110, bottom_y - 285, 125, 48, "RoPE", fill=orange, stroke=orange)
    _shape(out, x + 300, bottom_y - 285, 125, 48, "Cat", fill=orange, stroke=orange)
    _shape(out, x + 530, bottom_y - 285, 125, 48, "Split", fill=orange, stroke=orange)
    _shape(out, x + 705, bottom_y - 285, 125, 48, "Cat", fill=orange, stroke=orange)
    _shape(
        out, x + 325, bottom_y - 375, 460, 58, "Attention", fill=orange, stroke=orange
    )
    _text(out, "Q", x + 210, bottom_y - 316, size=22, fill=red, weight=700)
    _text(out, "K", x + 565, bottom_y - 316, size=22, fill=red, weight=700)
    _text(out, "V", x + 745, bottom_y - 316, size=22, fill=red, weight=700)
    _shape(
        out,
        x + 455,
        y + 74,
        230,
        54,
        "Linear(O)",
        fill=orange,
        stroke=orange,
    )
    _shape(out, x + 705, bottom_y - 172, 120, 48, "kv cache", fill=red, stroke=red)
    _shape(out, x + 355, bottom_y - 172, 120, 48, "pe cache", fill=red, stroke=red)

    _arrow(out, x + w // 2, y + h - 10, x + 220, bottom_y + 54)
    _arrow(out, x + w // 2, y + h - 10, x + 620, bottom_y + 54)
    _arrow(out, x + 220, bottom_y, x + 240, bottom_y - 57)
    _arrow(out, x + 620, bottom_y, x + 620, bottom_y - 57)
    _arrow(out, x + 240, bottom_y - 105, x + 250, bottom_y - 150)
    _arrow(out, x + 620, bottom_y - 105, x + 620, bottom_y - 150)
    _arrow(out, x + 250, bottom_y - 200, x + 180, bottom_y - 237)
    _arrow(out, x + 250, bottom_y - 200, x + 350, bottom_y - 237)
    _arrow(out, x + 620, bottom_y - 200, x + 592, bottom_y - 237)
    _arrow(out, x + 620, bottom_y - 200, x + 745, bottom_y - 237)
    _arrow(out, x + 180, bottom_y - 285, x + 365, bottom_y - 317)
    _arrow(out, x + 365, bottom_y - 285, x + 470, bottom_y - 317)
    _arrow(out, x + 592, bottom_y - 285, x + 575, bottom_y - 317)
    _arrow(out, x + 745, bottom_y - 285, x + 710, bottom_y - 317)
    _arrow(out, x + 555, bottom_y - 375, x + 570, y + 128)
    _shape_label(out, "[seq_len, hidden_size]", x + 402, y + h - 20)
    _shape_label(out, "[seq_len, heads, qk_head_dim]", x + 70, bottom_y - 215)
    _shape_label(out, "[seq_len, kv_lora_rank]", x + 660, bottom_y - 104)


def _draw_router_panel(out: list[str], x: int, y: int, w: int, h: int) -> None:
    teal = "#4CC9C0"
    _panel(out, x, y, w, h, "Router (Gate)", teal, font_size=28)
    _shape(out, x + 80, y + h - 72, 220, 52, "Linear", fill=teal, stroke=teal)
    _shape(
        out, x + 80, y + h - 158, 220, 52, "Sigmoid", fill="#FB7573", stroke="#FB7573"
    )
    _shape(out, x + 60, y + 78, 125, 56, "Top k", fill=teal, stroke=teal)
    _shape(out, x + 215, y + 78, 145, 56, "Select / Sum", fill=teal, stroke=teal)
    _arrow(out, x + 190, y + h - 20, x + 190, y + h - 106)
    _arrow(out, x + 190, y + h - 158, x + 128, y + 134)
    _arrow(out, x + 190, y + h - 158, x + 287, y + 134)
    _arrow(out, x + 185, y + 106, x + 215, y + 106)
    _shape_label(out, "[seq_len, num_experts]", x + 70, y + h - 174)
    _shape_label(out, "top_k_index   top_k_weights", x + 40, y + 58)


def _draw_index_panel(out: list[str], x: int, y: int, w: int, h: int) -> None:
    blue = "#84B9FF"
    _panel(out, x, y, w, h, "Indexer", blue, font_size=26)
    _shape(
        out,
        x + 40,
        y + 78,
        w - 80,
        54,
        "Select(top k)",
        fill="#FFFFFF",
        stroke=blue,
        text_fill="#7288F4",
    )
    _shape(
        out,
        x + 58,
        y + 175,
        w - 116,
        62,
        "Sparse KV blocks",
        fill="#FFFFFF",
        stroke=blue,
        text_fill="#111111",
    )
    _arrow(out, x + w // 2, y + h - 24, x + w // 2, y + 237)
    _arrow(out, x + w // 2, y + 175, x + w // 2, y + 132)
    _shape_label(out, "index_topk / local + compressed blocks", x + 36, y + h - 42)


def _draw_vision_panel(
    out: list[str], x: int, y: int, w: int, h: int, *, moonvit: bool = False
) -> None:
    teal = "#4CC9C0"
    blue = "#3B82F6"
    _panel(out, x, y, w, h, "Vision path", teal, font_size=30)
    _shape(out, x + 58, y + h - 84, 240, 56, "Image / video", fill=teal, stroke=teal)
    _shape(
        out,
        x + 355,
        y + h - 84,
        250,
        56,
        "Text tokens",
        fill="#7288F4",
        stroke=blue,
    )
    _shape(
        out,
        x + 58,
        y + h - 190,
        240,
        58,
        "MoonViT" if moonvit else "ViT / SigLIP",
        fill=teal,
        stroke=teal,
    )
    _shape(
        out,
        x + 58,
        y + h - 300,
        240,
        58,
        "Projector",
        fill=teal,
        stroke=teal,
    )
    _shape(
        out,
        x + 355,
        y + h - 300,
        250,
        58,
        "LLM Decoder",
        fill="#7288F4",
        stroke=blue,
    )
    _arrow(out, x + 178, y + h - 84, x + 178, y + h - 132)
    _arrow(out, x + 178, y + h - 190, x + 178, y + h - 242)
    _arrow(out, x + 298, y + h - 271, x + 355, y + h - 271)
    _arrow(out, x + 480, y + h - 84, x + 480, y + h - 242)
    _shape_label(out, "patch merge / pixel shuffle / mm projector", x + 52, y + 60)


def _draw_dit_panel(
    out: list[str], x: int, y: int, w: int, h: int, *, bimodal: bool = False
) -> None:
    orange = "#FF9F1A"
    purple = "#7E2E9E"
    _panel(out, x, y, w, h, "DiT denoiser", orange, font_size=30)
    _shape(
        out, x + 70, y + h - 76, 230, 54, "Latent tokens", fill=orange, stroke=orange
    )
    _shape(
        out,
        x + 380,
        y + h - 76,
        250,
        54,
        "Conditioning",
        fill="#7288F4",
        stroke="#3B82F6",
    )
    _shape(out, x + 70, y + h - 190, 230, 56, "Attention", fill=orange, stroke=orange)
    _shape(
        out,
        x + 380,
        y + h - 190,
        250,
        56,
        "MLP / Modulation",
        fill="#8DD348",
        stroke="#8DD348",
    )
    _shape(
        out,
        x + 210,
        y + 82,
        300,
        58,
        "Video tower + Audio tower" if bimodal else "VAE decode",
        fill=purple,
        stroke=purple,
    )
    _shape(out, x + w - 210, y + 82, 135, 58, "Cache", fill="#F00000", stroke="#F00000")
    _arrow(out, x + 185, y + h - 76, x + 185, y + h - 134)
    _arrow(out, x + 505, y + h - 76, x + 505, y + h - 134)
    _arrow(out, x + 185, y + h - 190, x + 360, y + 140)
    _arrow(out, x + 505, y + h - 190, x + 360, y + 140)
    _arrow(out, x + 510, y + 111, x + w - 210, y + 111)
    _shape_label(out, "[batch, frames, h*w, channels]", x + 60, y + h - 214)


def _draw_generic_details(
    out: list[str],
    template: Template,
    model: str,
    anchors: dict[str, tuple[int, int]],
) -> None:
    tid = template.template_id
    if tid in {"mla_moe", "moonvit_mla_moe"}:
        if tid == "moonvit_mla_moe":
            _draw_vision_panel(out, 600, 155, 590, 300, moonvit=True)
            _draw_moe_panel(out, 1230, 155, 560, 300, experts="256 experts")
            _draw_attention_panel(out, 600, 520, 920, 570, "MLA (MHA mode)")
            _draw_mlp_panel(out, 1535, 520, 330, 420, "MLP(FFN)")
            _arrow(
                out,
                anchors["input"][0],
                anchors["input"][1],
                600,
                730,
                dashed=True,
                stroke="#F59E0B",
            )
            return
        _draw_moe_panel(out, 610, 145, 680, 300, experts="routed experts")
        _draw_mlp_panel(out, 1330, 145, 470, 300, "MLP(FFN)")
        _draw_attention_panel(out, 610, 510, 920, 580, "MLA (MHA mode)")
        _arrow(
            out,
            anchors["ffn"][0],
            anchors["ffn"][1],
            610,
            280,
            dashed=True,
            stroke="#3B82F6",
        )
        _arrow(
            out,
            anchors["attention"][0],
            anchors["attention"][1],
            610,
            820,
            dashed=True,
            stroke="#F59E0B",
        )
        return

    if tid == "dsa_moe":
        _draw_moe_panel(out, 610, 145, 650, 295, experts="routed experts")
        _draw_mlp_panel(out, 1300, 145, 480, 295, "MLP(FFN)")
        _draw_attention_panel(out, 610, 510, 840, 560, "DSA (MQA mode)")
        _draw_index_panel(out, 1490, 610, 340, 330)
        _arrow(
            out,
            anchors["ffn"][0],
            anchors["ffn"][1],
            610,
            280,
            dashed=True,
            stroke="#3B82F6",
        )
        _arrow(
            out,
            anchors["attention"][0],
            anchors["attention"][1],
            610,
            820,
            dashed=True,
            stroke="#F59E0B",
        )
        return

    if tid in {"decoder_moe", "swa_moe_mtp"}:
        _draw_moe_panel(out, 610, 145, 650, 315, experts="routed experts")
        _draw_router_panel(out, 1300, 145, 430, 315)
        title = "SWA / GQA" if tid == "swa_moe_mtp" else "GQA / MQA Attention"
        _draw_attention_panel(out, 610, 530, 875, 540, title)
        _draw_mlp_panel(out, 1510, 535, 330, 410, "Expert MLP")
        if tid == "swa_moe_mtp":
            _shape(
                out,
                1270,
                470,
                230,
                56,
                "Attention sink",
                fill="#F00000",
                stroke="#F00000",
            )
        _arrow(
            out,
            anchors["ffn"][0],
            anchors["ffn"][1],
            610,
            305,
            dashed=True,
            stroke="#3B82F6",
        )
        _arrow(
            out,
            anchors["attention"][0],
            anchors["attention"][1],
            610,
            800,
            dashed=True,
            stroke="#F59E0B",
        )
        return

    if tid == "decoder_dense":
        _draw_attention_panel(out, 610, 185, 860, 570, "GQA / MHA Attention")
        _draw_mlp_panel(out, 610, 805, 650, 300, "Dense MLP")
        _arrow(
            out,
            anchors["attention"][0],
            anchors["attention"][1],
            610,
            520,
            dashed=True,
            stroke="#F59E0B",
        )
        _arrow(
            out,
            anchors["ffn"][0],
            anchors["ffn"][1],
            610,
            930,
            dashed=True,
            stroke="#8DD348",
        )
        return

    if tid == "hybrid_delta_moe":
        _panel(out, 610, 145, 625, 335, "Hybrid block cycle", "#F59E0B", font_size=30)
        _shape(
            out, 675, 365, 235, 56, "Gated DeltaNet", fill="#FF9F1A", stroke="#FF9F1A"
        )
        _shape(
            out, 945, 365, 220, 56, "Gated DeltaNet", fill="#FF9F1A", stroke="#FF9F1A"
        )
        _shape(
            out, 675, 245, 235, 56, "Gated DeltaNet", fill="#FF9F1A", stroke="#FF9F1A"
        )
        _shape(
            out, 945, 245, 220, 56, "Gated Attention", fill="#7E2E9E", stroke="#7E2E9E"
        )
        _arrow(out, 910, 393, 945, 393)
        _arrow(out, 1055, 365, 1055, 301)
        _arrow(out, 945, 273, 910, 273)
        _arrow(out, 792, 245, 792, 301)
        _draw_moe_panel(out, 1280, 145, 510, 335, experts="MoE / FFN")
        _draw_attention_panel(out, 610, 545, 900, 525, "Linear-state + gated attention")
        _shape(
            out, 1540, 600, 250, 64, "Recurrent state", fill="#F00000", stroke="#F00000"
        )
        _arrow(
            out,
            anchors["attention"][0],
            anchors["attention"][1],
            610,
            780,
            dashed=True,
            stroke="#F59E0B",
        )
        return

    if tid == "vlm":
        _draw_vision_panel(out, 610, 145, 640, 360)
        _draw_attention_panel(out, 610, 560, 850, 510, "LLM Attention")
        _draw_mlp_panel(out, 1490, 575, 340, 390, "LLM FFN / MoE")
        _arrow(
            out,
            anchors["input"][0],
            anchors["input"][1],
            610,
            360,
            dashed=True,
            stroke="#4CC9C0",
        )
        return

    if tid in {"diffusion_dit", "diffusion_moe_video", "mova_bimodal_dit"}:
        _draw_dit_panel(out, 610, 145, 770, 450, bimodal=tid == "mova_bimodal_dit")
        if tid == "diffusion_moe_video":
            _draw_moe_panel(
                out,
                1420,
                145,
                390,
                450,
                experts="timestep experts",
                router_fill="#4CC9C0",
            )
        else:
            _panel(out, 1420, 145, 390, 450, "Acceleration", "#F00000", font_size=28)
            _shape(
                out, 1485, 260, 260, 58, "TeaCache", fill="#F00000", stroke="#F00000"
            )
            _shape(
                out, 1485, 365, 260, 58, "CUDA Graph", fill="#F00000", stroke="#F00000"
            )
        _draw_attention_panel(out, 610, 660, 870, 410, "DiT block detail")
        _arrow(
            out,
            anchors["attention"][0],
            anchors["attention"][1],
            610,
            820,
            dashed=True,
            stroke="#F59E0B",
        )
        return

    if tid == "llada":
        _draw_attention_panel(
            out, 610, 160, 840, 520, "Bidirectional denoising Transformer"
        )
        _panel(out, 1490, 160, 340, 520, "Mask schedule", "#F00000", font_size=28)
        _shape(
            out, 1540, 300, 235, 58, "Mask predictor", fill="#F00000", stroke="#F00000"
        )
        _shape(
            out, 1540, 425, 235, 58, "Refine tokens", fill="#7288F4", stroke="#3B82F6"
        )
        _draw_mlp_panel(out, 610, 745, 600, 300, "MLP")
        return

    if tid == "reranker":
        _panel(out, 610, 170, 760, 390, "Encoder reranker", "#7288F4", font_size=30)
        _shape(out, 680, 430, 240, 58, "Query tokens", fill="#7288F4", stroke="#3B82F6")
        _shape(
            out, 1010, 430, 240, 58, "Document tokens", fill="#7288F4", stroke="#3B82F6"
        )
        _shape(
            out,
            800,
            300,
            330,
            60,
            "Bidirectional encoder",
            fill="#FF9F1A",
            stroke="#FF9F1A",
        )
        _shape(out, 850, 190, 230, 58, "Score head", fill="#8DD348", stroke="#8DD348")
        _arrow(out, 800, 430, 900, 360)
        _arrow(out, 1130, 430, 1030, 360)
        _arrow(out, 965, 300, 965, 248)
        _draw_mlp_panel(out, 610, 645, 600, 330, "Classifier MLP")
        return

    if tid in {"tts_dual_ar", "audio_hybrid_serving"}:
        _panel(
            out, 610, 150, 760, 420, "Audio generation stack", "#4CC9C0", font_size=30
        )
        _shape(
            out,
            690,
            445,
            230,
            58,
            "Text / reference audio",
            fill="#4CC9C0",
            stroke="#4CC9C0",
        )
        _shape(out, 1000, 445, 230, 58, "RVQ / codec", fill="#FF9F1A", stroke="#FF9F1A")
        _shape(out, 690, 320, 230, 58, "Slow AR", fill="#7288F4", stroke="#3B82F6")
        _shape(out, 1000, 320, 230, 58, "Fast AR", fill="#7288F4", stroke="#3B82F6")
        _shape(out, 845, 200, 230, 58, "DAC vocoder", fill="#8DD348", stroke="#8DD348")
        _arrow(out, 805, 445, 805, 378)
        _arrow(out, 1115, 445, 1115, 378)
        _arrow(out, 920, 320, 1000, 349)
        _arrow(out, 1115, 320, 960, 258)
        _draw_attention_panel(out, 610, 650, 820, 410, "Paged KV / audio state")
        _shape(
            out, 1470, 250, 300, 70, "SGLang engine", fill="#7288F4", stroke="#3B82F6"
        )
        _shape(
            out,
            1470,
            370,
            300,
            70,
            "Audio I/O server",
            fill="#4CC9C0",
            stroke="#4CC9C0",
        )
        return

    _draw_attention_panel(out, 610, 185, 860, 570, "Attention")
    _draw_mlp_panel(out, 610, 805, 650, 300, "FFN")


def svg_for(model: str, template: Template) -> str:
    width = 1900
    height = 1160
    primary = primary_style(model, template)
    title = f"{model} Architecture"
    title_font = max(30, min(38, int(2600 / max(len(title), 1))))
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,5 L0,10 z" fill="#111111"/></marker>',
        "</defs>",
        '<rect class="architecture-grid" width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="{width // 2}" y="66" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="{title_font}" font-weight="700" fill="#111111">{escape(title)}</text>',
        f'<text x="{width // 2}" y="106" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="22" fill="#111111">Generated in InfraTech-style layout from SGLang model code surfaces</text>',
        f'<text x="{width // 2}" y="136" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="16" fill="#555555">{escape(_source_note(template))}</text>',
    ]
    anchors = _draw_left_spine(out, model, template, primary)
    _draw_generic_details(out, template, model, anchors)
    _text(
        out,
        "context / vocab: read from concrete model config when refining",
        96,
        1135,
        size=17,
        weight=700,
    )
    _text(
        out,
        "Generated diagram: module-level; public originals are returned unchanged when available",
        1280,
        1135,
        size=14,
        fill="#666666",
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_generated(
    model: str, template: Template, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{slugify(model)}-{template.template_id}"
    svg_path = output_dir / f"{base}.svg"
    mermaid_path = output_dir / f"{base}.mmd"
    svg_path.write_text(svg_for(model, template), encoding="utf-8")
    mermaid_path.write_text(mermaid_for(model, template), encoding="utf-8")
    return svg_path, mermaid_path


def result_existing(query: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "existing",
        "query": query,
        "diagrams": [
            {
                "title": m["title"],
                "url": m["url"],
                "source": m["source"],
                "path": m["path"],
                "local_path": (
                    str(LOCAL_SOURCE_ROOTS[m["source"]] / m["path"])
                    if m["source"] in LOCAL_SOURCE_ROOTS
                    else None
                ),
            }
            for m in matches
        ],
    }


def result_generated(query: str, output_dir: Path) -> dict[str, Any]:
    template = infer_template(query)
    svg_path, mermaid_path = write_generated(query, template, output_dir)
    return {
        "kind": "generated",
        "query": query,
        "template": template.template_id,
        "description": template.description,
        "svg_path": str(svg_path),
        "mermaid_path": str(mermaid_path),
        "sglang_files": list(template.sglang_files),
        "cookbook_hints": list(template.cookbook_hints),
    }


def markdown(result: dict[str, Any]) -> str:
    if result["kind"] == "existing":
        lines = [f"## {result['query']} architecture diagrams", ""]
        for item in result["diagrams"]:
            lines.append(f"![{item['title']}]({item['url']})")
            lines.append("")
            lines.append(f"Source: {item['source']} `{item['path']}`")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    lines = [
        f"## {result['query']} generated architecture diagram",
        "",
        f"![{result['query']} architecture]({result['svg_path']})",
        "",
        f"Generated from `{result['template']}` template: {result['description']}",
        "",
        f"Mermaid source: `{result['mermaid_path']}`",
        "",
        "Inspect these SGLang source surfaces before refining:",
    ]
    lines.extend(f"- `{path}`" for path in result["sglang_files"])
    lines.append("")
    lines.append("Relevant sgl-cookbook hints:")
    lines.extend(f"- `{path}`" for path in result["cookbook_hints"])
    return "\n".join(lines).rstrip() + "\n"


def list_known(index: dict[str, Any]) -> str:
    lines = []
    for entry in sorted(index["diagrams"], key=lambda e: (e["source"], e["title"])):
        aliases = ", ".join(entry.get("aliases", [])[:5])
        lines.append(f"- {entry['title']} [{entry['source']}]: {aliases}")
    return "\n".join(lines) + "\n"


def first_heading(path: Path) -> str:
    in_fence = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if stripped.startswith("# ") and not stripped.startswith("##"):
                heading = stripped.lstrip("#").strip()
                if normalize(heading) not in {
                    "introduction",
                    "model introduction",
                    "1. model introduction",
                }:
                    return heading
    return path.stem.replace("_", ".")


def model_paths_from_doc(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    found: list[str] = []
    for match in MODEL_PATH_RE.finditer(text):
        value = next(group for group in match.groups() if group)
        if value not in found:
            found.append(value)
    return found


def cookbook_docs(root: Path) -> list[Path]:
    docs_root = root / "docs"
    docs: list[Path] = []
    for section in MODEL_DOC_DIRS:
        section_root = docs_root / section
        if not section_root.exists():
            continue
        for path in section_root.rglob("*.md"):
            if path.name.lower() == "readme.md":
                continue
            docs.append(path)
    return sorted(docs, key=lambda p: str(p.relative_to(root)))


def batch_gallery_markdown(items: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [
        "# sgl-cookbook Model Architecture Diagram Audit",
        "",
        f"Model documents: {len(items)}",
        "",
    ]
    for item in items:
        lines.extend([f"## {item['query']}", "", f"Cookbook: `{item['doc']}`", ""])
        if item["kind"] == "existing":
            for diagram in item["diagrams"]:
                image_path = diagram.get("local_path") or diagram["url"]
                lines.append(f"![{diagram['title']}]({image_path})")
                lines.append("")
                lines.append(f"Source: {diagram['source']} `{diagram['path']}`")
                lines.append("")
        else:
            lines.append(f"![{item['query']} architecture]({item['svg_path']})")
            lines.append("")
            lines.append(
                f"Generated from `{item['template']}` template: {item['description']}"
            )
            lines.append("")
    rel = output_dir / "manifest.json"
    lines.append(f"Manifest: `{rel}`")
    return "\n".join(lines).rstrip() + "\n"


def batch_gallery_html(items: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for item in items:
        if item["kind"] == "existing":
            media = "".join(
                f'<figure><img src="{escape(diagram.get("local_path") or diagram["url"])}" '
                f'alt="{escape(diagram["title"])}"><figcaption>'
                f'{escape(diagram["source"])}: {escape(diagram["path"])}</figcaption></figure>'
                for diagram in item["diagrams"]
            )
            badge = "existing"
        else:
            media = (
                f'<figure><img src="{escape(item["svg_path"])}" '
                f'alt="{escape(item["query"])} architecture"><figcaption>'
                f'generated template: {escape(item["template"])}</figcaption></figure>'
            )
            badge = "generated"
        cards.append(
            "<section>"
            f"<h2>{escape(item['query'])}</h2>"
            f"<p><span>{badge}</span> <code>{escape(item['doc'])}</code></p>"
            f"{media}"
            "</section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>sgl-cookbook architecture audit</title>"
        "<style>"
        "body{font-family:Inter,Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}"
        "section{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:14px}"
        "h1{margin:0 0 20px;font-size:26px}h2{font-size:18px;margin:0 0 8px}"
        "p{margin:0 0 10px;color:#475569}span{font-weight:700;color:#0369a1}"
        "figure{margin:0 0 10px}img{width:100%;height:auto;border:1px solid #e2e8f0;background:#fff}"
        "figcaption{font-size:12px;color:#64748b;margin-top:6px;word-break:break-word}"
        "code{font-size:12px}"
        "</style></head><body>"
        f"<h1>sgl-cookbook architecture audit ({len(items)} models)</h1>"
        f"<main>{''.join(cards)}</main></body></html>"
    )


def batch_sgl_cookbook(
    root: Path, output_dir: Path, max_existing: int
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for doc in cookbook_docs(root):
        title = first_heading(doc)
        rel_doc = str(doc.relative_to(root))
        candidates = [
            title,
            doc.stem.replace("_", "."),
            doc.stem.replace("_", "-"),
            *model_paths_from_doc(doc),
        ]
        matches = find_existing_any(candidates, max_existing)
        result = (
            result_existing(title, matches)
            if matches
            else result_generated(title, output_dir / "generated")
        )
        result["doc"] = rel_doc
        result["candidate_queries"] = candidates
        items.append(result)

    summary = {
        "model_count": len(items),
        "existing_count": sum(1 for item in items if item["kind"] == "existing"),
        "generated_count": sum(1 for item in items if item["kind"] == "generated"),
    }
    manifest = {
        "cookbook_root": str(root),
        "output_dir": str(output_dir),
        "summary": summary,
        "items": items,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gallery.md").write_text(
        batch_gallery_markdown(items, output_dir), encoding="utf-8"
    )
    (output_dir / "gallery.html").write_text(
        batch_gallery_html(items), encoding="utf-8"
    )
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="Model name or Hugging Face model id")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--force-generate", action="store_true", help="Ignore original diagram matches"
    )
    parser.add_argument("--max-existing", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--list-known", action="store_true")
    parser.add_argument(
        "--batch-sgl-cookbook",
        action="store_true",
        help="Generate a manifest and gallery for every sgl-cookbook model doc.",
    )
    parser.add_argument("--cookbook-root", type=Path, default=DEFAULT_COOKBOOK_ROOT)
    args = parser.parse_args(argv)

    index = load_index()
    if args.batch_sgl_cookbook:
        result = batch_sgl_cookbook(
            args.cookbook_root, args.output_dir, args.max_existing
        )
        if args.format == "json":
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        else:
            summary = result["summary"]
            sys.stdout.write(
                "Generated sgl-cookbook architecture audit at "
                f"`{args.output_dir}`: {summary['model_count']} models, "
                f"{summary['existing_count']} existing matches, "
                f"{summary['generated_count']} generated diagrams.\n"
            )
        return 0

    if args.list_known:
        sys.stdout.write(list_known(index))
        return 0

    if not args.model:
        parser.error("model is required unless --list-known is used")

    matches = (
        [] if args.force_generate else find_existing(args.model, args.max_existing)
    )
    result = (
        result_existing(args.model, matches)
        if matches
        else result_generated(args.model, args.output_dir)
    )

    if args.format == "json":
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
