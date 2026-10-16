import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "sglang-explainer-video" / "scripts"
spec = importlib.util.spec_from_file_location("pronounce", SCRIPTS / "pronounce.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def spoken(text):
    return P.to_spoken(text)[0]


def test_identifiers_never_reach_the_voice_with_underscores():
    s = spoken("用 flashmla_sparse_q8 做 prefill，打开 SGLANG_DSA_FUSE_TOPK 之后。")
    assert "_" not in s
    assert "Flash MLA sparse Q8" in s
    assert "SGLang DSA fuse top k" in s
    assert spoken("prepare_mtp_seed_indices") == "prepare MTP seed indices"
    assert spoken("DRAFT_EXTEND_V2") == "draft extend V two"


def test_lexicon_and_patterns():
    assert spoken("FlashInfer 和 TensorRT-LLM") == "Flash Infer 和 Tensor RT LLM"
    assert spoken("VAE 与 DiT 的 GEMM") == "V A E 与 D I T 的 G E M M"
    assert spoken("8×B300 上提速 3.7×，见 #42565。") == "8卡B300 上提速 3.7倍，见 42565。"
    assert spoken("吞吐 +5%，延迟 −12%") == "吞吐提升5%，延迟降低12%"
    assert spoken("fused top-k 的 top-K、bit-exact") == "fused top k 的 top K、bit exact"
    assert spoken("flashinfer 和 8GB") == "Flash Infer 和 8 G B"
    assert spoken("种子 5、1、-1，越界的 −2") == "种子 5、1、负1，越界的负2"
    assert spoken("1-3 步、GLM-5.3、x-1") == "1-3 步、GLM-5.3、x-1"


def test_terms_the_voice_reads_right_are_untouched():
    text = "SGLang 的 DSA 开了 fused 之后，PAGED 下标是 KV 池里的物理槽位，torch.compile 与 GLM-5.3。"
    assert spoken(text) == text


def test_no_space_between_han_characters():
    # a space between two Han characters makes the voice pause ~0.3 s
    assert spoken("种子里的 -1 是补位，查完再改回 -1；") == "种子里的负1 是补位，查完再改回负1；"


def test_map_back_restores_written_tokens_with_span_timing():
    s, spans = P.to_spoken("用 flashmla_sparse_q8 做 prefill。")
    said = ["用", "Flash", "MLA", "sparse", "Q", "8", "做", "prefill"]
    words = [{"text": t, "start": float(i), "end": i + 0.5} for i, t in enumerate(said)]
    out = P.map_back(s, spans, words)
    assert [w["text"] for w in out] == ["用", "flashmla_sparse_q8", "做", "prefill"]
    assert (out[1]["start"], out[1]["end"]) == (1.0, 5.5)


def test_map_back_cuts_a_word_that_straddles_a_span_edge():
    # edge-tts tokenizes "改回负1" as 改 | 回负 | 1: "负" belongs to the rewritten "-1"
    s, spans = P.to_spoken("查完再改回 -1；")
    words = [{"text": t, "start": a, "end": b} for t, a, b in
             [("查完", 0.0, 0.4), ("再", 0.4, 0.6), ("改", 0.6, 0.8), ("回负", 0.8, 1.2), ("1", 1.2, 1.5)]]
    out = P.map_back(s, spans, words)
    assert [w["text"] for w in out] == ["查完", "再", "改", "回", "-1"]
    assert (out[3]["start"], out[3]["end"]) == (0.8, 1.0)
    assert (out[4]["start"], out[4]["end"]) == (1.0, 1.5)


def test_user_picked_brand_readings():
    # chosen by ear on 2026-10-17: Qwen → Kwen (also inside Qwen3-8B), EAGLE → Eagle, RAGGED → Ragged
    assert spoken("Qwen3-8B 和 Qwen-Image") == "Kwen3-8B 和 Kwen-Image"
    assert spoken("EAGLE 三步，RAGGED 下标") == "Eagle 三步，Ragged 下标"
    assert spoken("Kimi K3 的 DeepSelect") == "Kimi K3 的 Deep Select"
