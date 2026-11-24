\
\
\
\
\
   

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def _pick_heads(hidden_size, requested):
    requested = max(1, min(int(requested), int(hidden_size)))
    for heads in range(requested, 0, -1):
        if hidden_size % heads == 0:
            return heads
    return 1

class KANLinear(nn.Module):
\
\
\
\
       

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        step = 2.0 / self.grid_size
        knots = torch.arange(
            -self.spline_order, self.grid_size + self.spline_order + 1, dtype=torch.float32
        )
        grid = knots * step - 1.0
        self.register_buffer("grid", grid)
        n_basis = self.grid_size + self.spline_order
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, n_basis)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.02)

    def _basis(self, x):
                                                   
        grid = self.grid
        x_exp = x.unsqueeze(-1)
        bases = ((x_exp >= grid[:-1]) & (x_exp < grid[1:])).to(dtype=x.dtype)
        for k in range(1, self.spline_order + 1):
            left_den = (grid[k:-1] - grid[: -(k + 1)]).clamp_min(1e-6)
            right_den = (grid[k + 1 :] - grid[1:(-k)]).clamp_min(1e-6)
            left = (x_exp - grid[: -(k + 1)]) / left_den
            right = (grid[k + 1 :] - x_exp) / right_den
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x):
        x_n = torch.tanh(x)
        base = F.linear(F.silu(x_n), self.base_weight)
        basis = self._basis(x_n)
        spline = torch.einsum("...ic,oic->...o", basis, self.spline_weight)
        return base + spline

class KANFusionBranch(nn.Module):
\
\
\
\
\
\
       

    def __init__(
        self,
        in_dim,
        d_model,
        num_nodes,
        num_steps,
        grid_size=5,
        spline_order=3,
        num_heads=4,
    ):
        super(KANFusionBranch, self).__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.num_nodes = int(num_nodes)
        self.num_steps = int(num_steps)
        heads = _pick_heads(self.in_dim, num_heads)
        self.num_heads = heads

        self.phi_t = KANLinear(self.in_dim, self.in_dim, grid_size, spline_order)
        self.phi_s = KANLinear(self.in_dim, self.in_dim, grid_size, spline_order)
        self.node_logit = nn.Parameter(torch.zeros(self.num_nodes))
        self.cross_t = nn.MultiheadAttention(
            self.in_dim, heads, dropout=0.0, batch_first=True
        )
        self.cross_s = nn.MultiheadAttention(
            self.in_dim, heads, dropout=0.0, batch_first=True
        )
        self.gate_t = nn.Linear(self.in_dim * 2, self.in_dim)
        self.gate_s = nn.Linear(self.in_dim * 2, self.in_dim)
        nn.init.zeros_(self.gate_t.weight)
        nn.init.zeros_(self.gate_t.bias)
        nn.init.zeros_(self.gate_s.weight)
        nn.init.zeros_(self.gate_s.bias)
        self.time_mix_t = nn.Linear(self.num_steps, 1)
        self.time_mix_s = nn.Linear(self.num_steps, 1)
        self.out_proj = nn.Linear(self.in_dim * 2, self.d_model)
        self.gamma_kan = nn.Parameter(torch.zeros(1))
        self.gamma_cross = nn.Parameter(torch.zeros(1))

    def _pool(self, h, node_logit=None):
                                                               
        logit = self.node_logit if node_logit is None else node_logit
        if logit.numel() != h.size(1):
            raise ValueError("node pool length %d does not match N=%d" % (logit.numel(), h.size(1)))
        weight = torch.softmax(logit, dim=0)
        pooled = torch.einsum("n,bntc->btc", weight, h)
        return pooled.mean(dim=0)

    def export_summary(self, h_t, h_s, node_logit=None):
                                                                               
        return {
            "temporal": self._pool(h_t, node_logit),
            "spatial": self._pool(h_s, node_logit),
        }

    def _memory(self, other_modes, key, like):
        if not other_modes:
            return None
        if isinstance(other_modes, dict):
            mem = other_modes.get(key)
            if mem is None:
                return None
            rows = mem
        else:
            rows = [item[key] for item in other_modes]
            if len(rows) == 0:
                return None
            rows = torch.stack(rows, dim=0)
        mem = rows.to(device=like.device, dtype=torch.float32).detach()
        if mem.dim() == 2:
            mem = mem.unsqueeze(0)
        if mem.dim() != 3:
            raise ValueError("KAN other-mode summary must be (T, C) or (M, T, C)")
        if mem.size(-1) != like.size(-1) or mem.size(-2) != like.size(-2):
            raise ValueError(
                "KAN other-mode summary shape %s does not match (T=%d, C=%d)"
                % (tuple(mem.shape), like.size(-2), like.size(-1))
            )
        if mem.size(0) == 0:
            return None
        return mem

    def _attend(self, h, mem, attn):
        batch = h.size(0)
        channels = h.size(-1)
        query = h.reshape(batch, -1, channels)
        key = mem.reshape(1, -1, channels).expand(batch, -1, -1)
        out, _ = attn(query, key, key, need_weights=False)
        return out.view_as(h)

    def _transfer(self, h, mem, phi, attn, gate):
        if mem is None:
            return torch.zeros_like(h)
        context = self._attend(h, mem, attn)
        pooled = torch.cat([h, context], dim=-1).mean(dim=(1, 2))
        channel_gate = torch.sigmoid(gate(pooled))
        transferred = channel_gate[:, None, None, :] * phi(context)
        return self.gamma_cross * transferred

    def _to_nodes(self, z, time_mix):
                                   
        mixed = time_mix(z.transpose(-1, -2)).squeeze(-1)
        return mixed

    def forward(self, h_t, h_s, other_modes=None):
        if h_t.size(-2) != self.num_steps or h_s.size(-2) != self.num_steps:
            raise ValueError(
                "KAN branch expected T=%d, got temporal %d and spatial %d"
                % (self.num_steps, h_t.size(-2), h_s.size(-2))
            )
        h_t = h_t.float()
        h_s = h_s.float()
        mem_t = self._memory(other_modes, "temporal", h_t)
        mem_s = self._memory(other_modes, "spatial", h_s)

        z_t = self.phi_t(h_t) + self._transfer(h_t, mem_t, self.phi_t, self.cross_t, self.gate_t)
        z_s = self.phi_s(h_s) + self._transfer(h_s, mem_s, self.phi_s, self.cross_s, self.gate_s)
        node_t = self._to_nodes(z_t, self.time_mix_t)
        node_s = self._to_nodes(z_s, self.time_mix_s)
        delta = self.out_proj(torch.cat([node_t, node_s], dim=-1))
        return self.gamma_kan * delta
