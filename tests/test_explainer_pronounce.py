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
    assert spoken("种子 5、1、-1，越界的 −2") == "种子 5、 1、负1，越界的负2"
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


def test_line_sense_hang_is_respelled_as_an_unambiguous_homophone():
    # 行 meaning "row / line" must be háng; the voice read 一行 as yì xíng in 31 of 74 past lines
    # (acoustic probe, 2026-10-19). 航 is always háng; captions keep 行.
    assert spoken("改一行代码就够了。") == "改一航代码就够了。"
    assert spoken("还是那一行，每一行都算。") == "还是那一航，每一航都算。"
    assert spoken("一个 CTA 一行，一行的上下文是 1000。") == "一个 CTA 一航，一航的上下文是 1000。"
    assert spoken("每行、整行、几行、两行、六行、七万八千零八十行、长行和短行") == \
        "每航、整航、几航、两航、六航、七万八千零八十航、长航和短航"
    assert spoken("第 3 行到第4行") == "第 3 航到第4航"
    assert spoken("一行行地扫，每行使用一个 warp，每行进行归约，一行程序") == \
        "一航航地扫，每航使用一个 warp，每航进行归约，一航程序"


def test_xing_words_are_left_alone():
    for text in ["程序运行很快，并行执行。", "统一行为，统一行动。", "唯一行得通的办法。", "万一行不通。",
                 "他们一行人到了。", "每行驶一公里。", "一行李箱。", "这样就行。", "每行使权利。"]:
        assert "航" not in spoken(text), text


def test_map_back_keeps_written_hang():
    s, spans = P.to_spoken("改一行代码")
    words = [{"text": t, "start": a, "end": b} for t, a, b in [("改", 0.0, 0.2), ("一航", 0.2, 0.6), ("代码", 0.6, 1.0)]]
    out = P.map_back(s, spans, words)
    assert "".join(w["text"] for w in out) == "改一行代码"


def test_terms_from_the_24gb_video():
    # measured 2026-10-19: checkpoint -> ChatPoint, OOM -> Om, bare kitchen -> Tichen, fa said as a syllable
    assert spoken("直接映射 checkpoint 文件") == "直接映射 check point 文件"
    assert spoken("第一个请求就会 OOM。") == "第一个请求就会 O O M。"
    assert spoken("comfy-kitchen 的 int8_linear") == "Comfy Kitchen 的 INT8 linear"
    assert spoken("默认仍然是精确的 fa，") == "默认仍然是精确的 F A，"


def test_terms_from_the_k3_video():
    # measured 2026-10-19: bare MLA heard as MBA 3/3 (spaced 3/3 right), SiTU -> C2, kimi_k3.py -> Timmy K3PY
    assert spoken("24 层 MLA 的 KV") == "24 层 M L A 的 KV"
    assert spoken("Gated MLA 的输出门") == "Gated M L A 的输出门"
    assert spoken("激活函数 SiTU 的上界") == "激活函数 S I T U 的上界"
    assert spoken("4 乘 tanh 再乘 sigmoid") == "4 乘 tan h 再乘 sigmoid"
    assert spoken("模型在 kimi_k3.py 里") == "模型在 Kimi K3 点 P Y 里"


def test_comma_between_numbers_is_not_a_thousands_separator():
    # measured 2026-10-22: Edge returned "2，257" as one word, read 两千二百五十七 3/3; "2， 257" splits it
    assert spoken("间距是 2，257 会舍入成 256。") == "间距是 2， 257 会舍入成 256。"
    assert spoken("M 取 1、16、256；") == "M 取 1、 16、 256；"
    assert spoken("间距是二，257") == "间距是二，257"


def test_terms_from_the_dsv41_k3_sharing_video():
    # PR 42929 + 45126 video (2026-10-23): all-caps DSPARK is heard as "Spark";
    # DSpark passes. 行 after a K/M/G count ("262K 行") is line-sense too.
    assert spoken("4 张 B200、DSPARK、单请求") == "4 张 B200、DSpark、单请求"
    assert spoken("262K 行从 0.381 毫秒降到 0.089 毫秒；16K行太小") == "262K 航从 0.381 毫秒降到 0.089 毫秒；16K航太小"
    assert spoken("2M 行的数据") == "2M 航的数据"
    assert "航" not in spoken("SGLang 的 K 行为")


def test_terms_from_the_model_release_video():
    # Model-release video (2026-10-24): 长上下文 and 擅长 were zhǎng in 4/6 takes;
    # 常 (only ever cháng) fixes both (0/6). 变长 / 长程 / 长篇 were already right.
    assert spoken("RL 框架、长上下文推理") == "RL 框架、常上下文推理"
    assert spoken("它尤其擅长参数化建模") == "它尤其擅常参数化建模"
    assert spoken("不随上下文变长，长程任务，长篇讲解") == "不随上下文变长，长程任务，长篇讲解"
    assert "S W E" in spoken("SWE Marathon 51.7") and "Browse Comp" in spoken("BrowseComp 92.1")


def test_leading_and_trailing_underscores_stay_with_the_identifier():
    # PR 28185 video (2026-10-27): TOKEN started at a letter, so the "_" of a private
    # method stayed in the text and the voice read 「下划线 commit int8 checkpoint」.
    assert spoken("入口，_commit_int8_checkpoint：先申请") == "入口，commit INT8 checkpoint：先申请"
    assert spoken("池子在 _init_mamba_pool 里创建") == "池子在 init mamba pool 里创建"
    assert spoken("__init__ 和 x_y") == "init 和 x y"
    _, spans = P.to_spoken("新增的 _free_mamba_value，开了")
    assert [w for *_, w in spans] == ["_free_mamba_value"]


def test_ack_is_spelled():
    # PD security blog video (2026-10-27): ack / ACK were heard as at / act / App;
    # spelled A C K passes. ABORT_ACK needs a hyphenated spelling so the A does not
    # run into abort's final t.
    assert spoken("等所有 rank 的 ack 到齐") == "等所有 rank 的 A C K 到齐"
    assert spoken("回一个 ABORT_ACK。") == "回一个 Abort A-C-K。"
    assert spoken("单边 RDMA WRITE") == "单边 RDMA WRITE"
