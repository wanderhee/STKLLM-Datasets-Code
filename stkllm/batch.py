from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

def unpack_batch(batch_data) -> Tuple[Any, Any, Optional[List[str]]]:
    if len(batch_data) == 2:
        encoder_inputs, labels = batch_data
        return encoder_inputs, labels, None
    encoder_inputs, labels, prompts = batch_data
    if isinstance(prompts, (list, tuple)):
        prompt_list = ["" if p is None else str(p) for p in prompts]
    else:
        prompt_list = ["" if prompts is None else str(prompts)]
    return encoder_inputs, labels, prompt_list

def forward_with_prompts(net, encoder_inputs, event_prompts: Optional[Sequence[str]] = None):
    if event_prompts is None:
        return net(encoder_inputs)
    return net(encoder_inputs, event_prompts=event_prompts)
