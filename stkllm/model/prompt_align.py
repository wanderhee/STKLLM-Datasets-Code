\
\
\
   

import torch
import torch.nn as nn

def _pick_heads(hidden_size, requested):
    requested = max(1, min(int(requested), int(hidden_size)))
    for heads in range(requested, 0, -1):
        if hidden_size % heads == 0:
            return heads
    return 1

class PromptSTAlign(nn.Module):
\
\
\
\
\
\
\
       

    def __init__(self, hidden_size, num_heads=8, dropout=0.0):
        super(PromptSTAlign, self).__init__()
        heads = _pick_heads(hidden_size, num_heads)
        self.hidden_size = int(hidden_size)
        self.num_heads = heads
        self.prompt_from_nodes = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.nodes_from_prompt = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.norm_p = nn.LayerNorm(hidden_size)
        self.norm_m = nn.LayerNorm(hidden_size)
        self.gamma_p = nn.Parameter(torch.zeros(1))
        self.gamma_m = nn.Parameter(torch.zeros(1))

    def forward(self, prompt_embeds, node_feats, prompt_attention_mask):
\
\
\
\
           
        dtype = prompt_embeds.dtype
        device = prompt_embeds.device
        prompt = prompt_embeds.float()
        nodes = node_feats.float()

        delta_p, _ = self.prompt_from_nodes(
            prompt, nodes, nodes, need_weights=False
        )

        pad = (prompt_attention_mask == 0).to(device=device)
        all_pad = pad.all(dim=-1)
        if all_pad.any():
            pad = pad.clone()
            pad[all_pad, 0] = False
        delta_m, _ = self.nodes_from_prompt(
            nodes,
            prompt,
            prompt,
            key_padding_mask=pad,
            need_weights=False,
        )
        if all_pad.any():
            delta_m = delta_m.masked_fill(all_pad.view(-1, 1, 1), 0.0)

        valid = prompt_attention_mask.to(device=device, dtype=delta_p.dtype).unsqueeze(-1)
        delta_p = self.norm_p(delta_p) * valid
        delta_m = self.norm_m(delta_m)

        prompt_out = prompt_embeds + self.gamma_p.to(dtype=dtype, device=device) * delta_p.to(dtype)
        node_out = node_feats + self.gamma_m.to(dtype=dtype, device=device) * delta_m.to(dtype)
        return prompt_out, node_out
