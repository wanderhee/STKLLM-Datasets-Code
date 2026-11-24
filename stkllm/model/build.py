import torch

from stkllm.model.kan import KANFusionBranch
from stkllm.model.backbone import make_STATPLLM_model as _make_STATPLLM_model
from stkllm.model.prompt_align import PromptSTAlign

def attach_prompt_st_align(net, num_heads=8, dropout=0.0):
    block = net.llama3_lora
    align = PromptSTAlign(block.hidden_size, num_heads=num_heads, dropout=dropout)
    align.to(device=block.llama_input_device, dtype=torch.float32)
    block.prompt_st_align = align

    def _align_prompt_and_nodes(prompt_embeds, node_feats, prompt_attention_mask):
        return block.prompt_st_align(prompt_embeds, node_feats, prompt_attention_mask)

    block._align_prompt_and_nodes = _align_prompt_and_nodes
    return net

def attach_kan_branch(net, grid_size=5, spline_order=3, num_heads=4):
    _, channels, num_nodes, num_steps = net.b1.shape
    d_model = net.input_conv.out_channels
    branch = KANFusionBranch(
        in_dim=int(channels),
        d_model=int(d_model),
        num_nodes=int(num_nodes),
        num_steps=int(num_steps),
        grid_size=grid_size,
        spline_order=spline_order,
        num_heads=num_heads,
    )
    branch.to(device=net.compute_device, dtype=torch.float32)
    net.kan_branch = branch

    def _kan_residual(x_in, h_t, h_s, other_modes=None):
        delta = branch(h_t, h_s, other_modes=other_modes)
        return x_in + delta.to(dtype=x_in.dtype)

    net._kan_residual = _kan_residual
    return net

def make_stkllm(
    *args,
    prompt_st_heads=8,
    prompt_st_dropout=0.0,
    kan_grid_size=5,
    kan_spline_order=3,
    kan_heads=4,
    **kwargs,
):
    use_qlora = kwargs.pop("use_qlora", True)
    net = _make_STATPLLM_model(*args, use_qlora=use_qlora, **kwargs)
    if net.llama3_lora is None:
        return net
    net = attach_prompt_st_align(
        net, num_heads=prompt_st_heads, dropout=prompt_st_dropout
    )
    return attach_kan_branch(
        net,
        grid_size=kan_grid_size,
        spline_order=kan_spline_order,
        num_heads=kan_heads,
    )
