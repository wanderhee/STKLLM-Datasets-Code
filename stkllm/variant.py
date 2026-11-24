from __future__ import annotations

from configparser import ConfigParser

_VALID_OUT_HEADS = ("linear", "mlp", "residual")

def read_out_head(config: ConfigParser) -> str:
                                                                    
    if config.has_section("Variant"):
        value = config["Variant"].get("out_head", "linear")
    else:
        value = "linear"
    out_head = str(value).strip().lower()
    if out_head not in _VALID_OUT_HEADS:
        raise ValueError("Variant.out_head must be one of %s; got %r" % (", ".join(_VALID_OUT_HEADS), value))
    return out_head
