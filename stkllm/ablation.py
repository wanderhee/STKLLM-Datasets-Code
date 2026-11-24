from __future__ import annotations

from configparser import ConfigParser
from typing import Tuple

def read_ablation_flags(config: ConfigParser) -> Tuple[bool, bool, bool]:
\
\
\
\
       
    if config.has_section("Ablation"):
        section = config["Ablation"]
        use_temporal = section.getboolean("use_temporal_encoder", fallback=True)
        use_spatial = section.getboolean("use_spatial_encoder", fallback=True)
        use_st_gated = section.getboolean("use_st_gated_residual", fallback=False)
        return use_temporal, use_spatial, use_st_gated
    return True, True, False
