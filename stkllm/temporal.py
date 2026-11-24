from __future__ import annotations

from configparser import ConfigParser
from typing import Any, Dict, Optional, Tuple

def compute_pfs_token_lengths(
    num_timesteps: int,
    patch_len: int,
    stride: int,
    merge_factor: int,
) -> Tuple[int, int]:
\
\
\
       
    T_eff = max(num_timesteps, patch_len)
    L = (T_eff - patch_len) // stride + 1
    pad_len = (merge_factor - L % merge_factor) % merge_factor
    L_pad = L + pad_len
    L_prime = L_pad // merge_factor
    return L, L_prime

def _get_section(config: ConfigParser) -> Optional[Any]:
    if config.has_section("TemporalEncoder"):
        return config["TemporalEncoder"]
    return None

def _parse_optional_int(value: str) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    return int(text)

def build_temporal_encoder_config(
    config: ConfigParser,
    len_input: int,
    in_channels: int,
    nb_chev_filter: int,
) -> Dict[str, Any]:
\
\
\
\
\
\
       
    section = _get_section(config)
    if section is None:
        return {"type": "gru"}

    encoder_type = section.get("temporal_encoder", "pfs").strip().lower()
    if encoder_type in ("gru", "legacy", "time_emb", "timeemb"):
        return {"type": "gru"}

    patch_len = int(section.get("pfs_patch_len", "4"))
    stride = int(section.get("pfs_stride", "2"))
    merge_factor = int(section.get("pfs_merge_factor", "2"))
    embed_dim = int(section.get("pfs_embed_dim", str(nb_chev_filter)))
    out_dim = int(section.get("pfs_out_dim", str(nb_chev_filter)))

    _, l_prime = compute_pfs_token_lengths(len_input, patch_len, stride, merge_factor)

    pfs_config = {
        "in_channels": in_channels,
        "patch_len": patch_len,
        "stride": stride,
        "embed_dim": embed_dim,
        "num_heads": int(section.get("pfs_num_heads", "4")),
        "max_period": int(section.get("pfs_max_period", str(len_input))),
        "min_period": int(section.get("pfs_min_period", "2")),
        "period_topk": int(section.get("pfs_period_topk", "3")),
        "fixed_period": _parse_optional_int(section.get("pfs_fixed_period", "")),
        "freq_topk": int(section.get("pfs_freq_topk", "4")),
        "merge_factor": merge_factor,
        "out_dim": out_dim,
        "dropout": float(section.get("pfs_dropout", "0.1")),
        "attn_dropout": float(section.get("pfs_attn_dropout", "0.1")),
        "use_causal_mask": section.getboolean("pfs_use_causal_mask", fallback=False),
        "share_attention_weights": section.getboolean("pfs_share_attention_weights", fallback=False),
    }

    return {
        "type": "pfs",
        "pfs": pfs_config,
        "pfs_token_len": l_prime,
    }
