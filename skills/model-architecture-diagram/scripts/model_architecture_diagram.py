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


def svg_for(model: str, template: Template) -> str:
    width = 980
    node_w = 520
    node_h = 78
    gap = 26
    x = (width - node_w) // 2
    title_h = 76
    height = title_h + len(template.nodes) * (node_h + gap) + 46
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="shadow" x="-5%" y="-5%" width="110%" height="120%"><feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#0F172A" flood-opacity="0.14"/></filter>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        f'<text x="{width // 2}" y="36" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="24" font-weight="700" fill="#111827">{escape(model)} architecture</text>',
        f'<text x="{width // 2}" y="60" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#475569">Generated fallback: {escape(template.description)}</text>',
    ]

    positions: dict[str, tuple[int, int]] = {}
    for i, node in enumerate(template.nodes):
        y = title_h + i * (node_h + gap)
        positions[node.key] = (x, y)
        fill, stroke = COLORS.get(node.kind, ("#FFFFFF", "#64748B"))
        out.append(
            f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2" filter="url(#shadow)"/>'
        )
        out.append(
            f'<text x="{x + 24}" y="{y + 30}" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#0F172A">{escape(node.title)}</text>'
        )
        for j, line in enumerate(wrap_svg_text(node.subtitle)):
            out.append(
                f'<text x="{x + 24}" y="{y + 54 + j * 15}" font-family="Inter,Arial,sans-serif" font-size="13" fill="#334155">{escape(line)}</text>'
            )

    for src, dst in template.edges:
        if src not in positions or dst not in positions:
            continue
        sx, sy = positions[src]
        dx, dy = positions[dst]
        x1 = sx + node_w // 2
        y1 = sy + node_h
        x2 = dx + node_w // 2
        y2 = dy
        out.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2 - 8}" stroke="#64748B" stroke-width="2"/>'
        )
        out.append(
            f'<polygon points="{x2 - 6},{y2 - 8} {x2 + 6},{y2 - 8} {x2},{y2}" fill="#64748B"/>'
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
