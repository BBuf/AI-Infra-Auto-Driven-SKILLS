"""Local HF config normalization for independently installed compute skill.

Alias contract mirrors pipeline model_profiles.normalize_model_config.
"""


def normalize_model_config(raw: dict) -> dict:
    """Flatten public text configs and normalize HF dimension aliases.

    Preserve the wrapper identity as outer_model_type for profile inference.
    Derived shared width is total width, not width per shared expert.
    """
    config = dict(raw)
    for key in ("text_config", "language_config", "llm_config"):
        inner = raw.get(key)
        if isinstance(inner, dict):
            config["outer_model_type"] = raw.get("model_type", "")
            config.update(inner)
            break
    if not config.get("num_hidden_layers"):
        config["num_hidden_layers"] = config.get("num_layers") or len(
            config.get("layers_block_type")
            or config.get("hybrid_override_pattern")
            or []
        )
    aliases = {
        "num_experts": ("n_routed_experts", "num_local_experts", "moe_num_experts"),
        "num_experts_per_tok": (
            "num_experts_per_token",
            "moe_topk",
            "moe_top_k",
            "top_k_experts",
        ),
        "num_shared_experts": ("n_shared_experts",),
        "routed_expert_intermediate_size": (
            "moe_intermediate_size",
            "expert_ffn_hidden_size",
        ),
        "shared_expert_intermediate_size": (
            "shared_intermediate_size",
            "moe_shared_expert_intermediate_size",
            "share_expert_dim",
        ),
    }
    for target, sources in aliases.items():
        if target not in config:
            for source in sources:
                if source in config:
                    config[target] = config[source]
                    break
    if "shared_expert_intermediate_size" not in config and "n_shared_experts" in config:
        config["shared_expert_intermediate_size"] = config.get(
            "moe_intermediate_size", 0
        ) * (config.get("n_shared_experts") or 0)
    config["moe"] = bool(config.get("moe") or (config.get("num_experts") or 0) > 0)
    config["mhc"] = bool(
        config.get("mhc") or config.get("hc_mult") or config.get("hc_count")
    )
    return config


def profile_name(raw):
    config = normalize_model_config(raw)
    types = {config.get("model_type", ""), config.get("outer_model_type", "")}
    identities = {
        "deepseek_v41": "dsv41",
        "deepseek_v41_text": "dsv41",
        "deepseek_v4": "dsv4_csa_hca",
        "kimi_k3": "kimi_k3",
        "glm5_next": "glm5_next",
        "glm5_next_text": "glm5_next",
        "hy_v4": "hy_v4",
        "nemotron_h": "nemotron_h",
        "qwen4_exp": "qwen4_exp",
        "qwen4_exp_text": "qwen4_exp",
        "glm_moe_dsa": "mla_dsa",
        "deepseek_v32": "mla_dsa",
    }
    if "kv_source_layer_ids" in config:
        return "dsv41"
    if "kimi_linear" in types and config.get("attn_res_block_size"):
        return "kimi_k3"
    for identity, name in identities.items():
        if identity in types:
            return name
    if config.get("compress_ratios"):
        return "dsv4_csa_hca"
    if config.get("index_topk") and config.get("kv_lora_rank"):
        return "mla_dsa"
    if config.get("kv_lora_rank"):
        return "dsv3_mla"
    return "generic_addnorm"
