import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyBiasedWindowAttention(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_period: int,
        freq_topk: int = 4,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ignore_dc: bool = True,
        use_causal_mask: bool = False,
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.max_period = max_period
        self.freq_topk = freq_topk
        self.ignore_dc = ignore_dc
        self.use_causal_mask = use_causal_mask

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

                                       
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros(2 * max_period - 1, num_heads)
        )

                                     
        self.win_bias_table = nn.Parameter(
            torch.zeros(max_period, max_period, num_heads)
        )

                                            
        self.freq_head_weight = nn.Parameter(
            torch.randn(num_heads, freq_topk) * 0.02
        )

                            
        self.freq_lambda = nn.Parameter(torch.tensor(1.0))

        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)
        nn.init.trunc_normal_(self.win_bias_table, std=0.02)

    def _position_bias(self, P: int, device: torch.device) -> torch.Tensor:
\
\
\
\
\
           
        if P > self.max_period:
            raise ValueError(
                                                                          
            )

        coords = torch.arange(P, device=device)
        rel = coords[:, None] - coords[None, :] + self.max_period - 1
        rel_bias = self.rel_pos_bias_table[rel]                     

        win_bias = self.win_bias_table[:P, :P, :]                   

        pos_bias = rel_bias + win_bias
        pos_bias = pos_bias.permute(2, 0, 1).contiguous()             

        return pos_bias

    def _frequency_differential_bias(
        self,
        x_windows: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        BN, M, P, d = x_windows.shape
        device = x_windows.device
        dtype = x_windows.dtype

        if self.freq_topk <= 0:
            return torch.zeros(
                BN, M, self.num_heads, P, P,
                device=device,
                dtype=dtype,
            )

                               
        if valid_mask is not None:
            x_for_fft = x_windows * valid_mask.unsqueeze(-1).to(dtype)
        else:
            x_for_fft = x_windows

                                           
                             
        freq = torch.fft.rfft(x_for_fft.float(), dim=2)

        num_freq = freq.size(2)

        if num_freq <= 1:
            return torch.zeros(
                BN, M, self.num_heads, P, P,
                device=device,
                dtype=dtype,
            )

                                               
        prev_freq = torch.zeros_like(freq)
        prev_freq[:, 1:, :, :] = freq[:, :-1, :, :]
        delta_freq = freq - prev_freq                                      

                                   
        drift_score = delta_freq.abs().mean(dim=-1)                      

        if self.ignore_dc:
                            
            score_for_topk = drift_score[:, :, 1:]                         
            available_freq = num_freq - 1
            k = min(self.freq_topk, available_freq)

            top_val, top_idx = torch.topk(score_for_topk, k=k, dim=-1)
            top_idx = top_idx + 1
        else:
            k = min(self.freq_topk, num_freq)
            top_val, top_idx = torch.topk(drift_score, k=k, dim=-1)

        if k == 0:
            return torch.zeros(
                BN, M, self.num_heads, P, P,
                device=device,
                dtype=dtype,
            )

                                            
                                  
        top_weight = torch.softmax(top_val, dim=-1)                      

                                      
                                                  
        pos = torch.arange(P, device=device)
        dist = pos[:, None] - pos[None, :]                           

                          
        freq_index = top_idx.float().unsqueeze(-1).unsqueeze(-1)

                          
        basis = torch.cos(
            2.0 * math.pi * freq_index * dist / float(P)
        )

                
        head_weight = self.freq_head_weight[:, :k]

                       
        head_freq_weight = (
            top_weight[:, :, None, :] * head_weight[None, None, :, :]
        )

                          
        b_freq = torch.einsum(
            "bmhk,bmkij->bmhij",
            head_freq_weight,
            basis,
        )

                                     
        b_freq = b_freq - b_freq.mean(dim=-1, keepdim=True)

        return b_freq.to(dtype)

    def forward(
        self,
        x_windows: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
\
\
\
\
\
\
\
           
        BN, M, P, d = x_windows.shape
        device = x_windows.device

        assert d == self.dim
        assert P <= self.max_period

                     
                          
        b_freq = self._frequency_differential_bias(
            x_windows,
            valid_mask=valid_mask,
        )

                
                   
        pos_bias = self._position_bias(P, device=device)

                         
        x = x_windows.reshape(BN * M, P, d)

                        
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            BN * M,
            P,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]
                                   

                                               
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                               

                                               
        b_freq = b_freq.reshape(BN * M, self.num_heads, P, P)
        attn = attn + pos_bias.unsqueeze(0)
        attn = attn + self.freq_lambda * b_freq

        if valid_mask is not None:
            key_mask = ~valid_mask.reshape(BN * M, P)
            key_mask = key_mask[:, None, None, :]                             
            attn = attn.masked_fill(key_mask, float("-inf"))

        if self.use_causal_mask:
            causal_mask = torch.triu(
                torch.ones(P, P, device=device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

                             
        out = torch.matmul(attn, v)                                             
        out = out.transpose(1, 2).reshape(BN * M, P, d)                     

        if valid_mask is not None:
            query_mask = valid_mask.reshape(BN * M, P).unsqueeze(-1)
            out = out * query_mask.to(out.dtype)

        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.reshape(BN, M, P, d)

        return out

class PFSAttnEncoder(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(
        self,
        in_channels: int,
        patch_len: int = 12,
        stride: int = 6,
        embed_dim: int = 64,
        num_heads: int = 4,
        max_period: int = 32,
        min_period: int = 2,
        period_topk: int = 3,
        fixed_period: Optional[int] = None,
        freq_topk: int = 4,
        merge_factor: int = 2,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_causal_mask: bool = False,
        share_attention_weights: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.max_period = max_period
        self.min_period = min_period
        self.period_topk = period_topk
        self.fixed_period = fixed_period

        self.merge_factor = merge_factor
        self.out_dim = out_dim if out_dim is not None else embed_dim
        self.eps = eps

                         
        self.patch_proj = nn.Linear(patch_len * in_channels, embed_dim)

                                         
        self.lsf_u = nn.Linear(embed_dim, embed_dim)
        self.lsf_g = nn.Linear(embed_dim * 2, embed_dim)
        self.lsf_norm = nn.LayerNorm(embed_dim)

                               
        self.window_attn = FrequencyBiasedWindowAttention(
            dim=embed_dim,
            num_heads=num_heads,
            max_period=max_period,
            freq_topk=freq_topk,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            use_causal_mask=use_causal_mask,
        )

                                       
        if share_attention_weights:
            self.shifted_window_attn = self.window_attn
        else:
            self.shifted_window_attn = FrequencyBiasedWindowAttention(
                dim=embed_dim,
                num_heads=num_heads,
                max_period=max_period,
                freq_topk=freq_topk,
                attn_dropout=attn_dropout,
                proj_dropout=dropout,
                use_causal_mask=use_causal_mask,
            )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.drop = nn.Dropout(dropout)

                                     
        self.merge_proj = nn.Linear(merge_factor * embed_dim, self.out_dim)
        self.out_norm = nn.LayerNorm(self.out_dim)

    def _instance_norm(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
\
\
\
\
\
\
\
           
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        sigma = torch.sqrt(var + self.eps)
        x_hat = (x - mu) / sigma
        return x_hat, mu, sigma

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
\
\
\
\
\
           
        BN, T, C = x.shape

        if T < self.patch_len:
            pad_len = self.patch_len - T
            x = F.pad(x, (0, 0, 0, pad_len))
            T = x.size(1)

        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )
                                        

        patches = patches.permute(0, 1, 3, 2).contiguous()
                                        

        patches = patches.reshape(BN, patches.size(1), self.patch_len * C)
                                       

        return patches

    def _light_stabilization_filter(self, e: torch.Tensor) -> torch.Tensor:
\
\
\
\
\
\
\
           
        delta = torch.zeros_like(e)
        delta[:, 1:, :] = e[:, 1:, :] - e[:, :-1, :]

        u = self.lsf_u(e)
        g = torch.sigmoid(self.lsf_g(torch.cat([e, delta], dim=-1)))

        r = e - g * u
        r = self.lsf_norm(r)

        return r

    def _select_period(self, r: torch.Tensor) -> int:
\
\
\
\
\
\
\
           
        BN, L, d = r.shape

        if self.fixed_period is not None:
            P = int(self.fixed_period)
            P = max(self.min_period, min(P, self.max_period, L))
            return P

        if L <= self.min_period:
            return max(1, L)

                                        
        with torch.no_grad():
            spectrum = torch.fft.rfft(r.detach().float(), dim=1)
            amp = spectrum.abs().mean(dim=(0, 2))            

            if amp.numel() <= 1:
                return min(self.max_period, max(self.min_period, L))

                                 
            amp[0] = 0.0

            k = min(self.period_topk, amp.numel() - 1)
            top_freq = torch.topk(amp[1:], k=k).indices + 1

            periods = torch.round(L / top_freq.float()).long()
            periods = torch.clamp(
                periods,
                min=self.min_period,
                max=min(self.max_period, L),
            )

                                          
            P = int(periods.min().item())

        P = max(self.min_period, min(P, self.max_period, L))

        return P

    def _pad_to_period(
        self,
        r: torch.Tensor,
        P: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
\
\
\
\
\
\
\
           
        BN, L, d = r.shape
        pad_len = (P - L % P) % P

        if pad_len > 0:
            r_pad = F.pad(r, (0, 0, 0, pad_len))
        else:
            r_pad = r

        L_pad = r_pad.size(1)

        valid = torch.ones(BN, L, device=r.device, dtype=torch.bool)
        if pad_len > 0:
            pad_valid = torch.zeros(BN, pad_len, device=r.device, dtype=torch.bool)
            valid = torch.cat([valid, pad_valid], dim=1)

        assert valid.size(1) == L_pad

        return r_pad, valid, pad_len

    def _partition_windows(
        self,
        r: torch.Tensor,
        P: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
\
\
\
\
\
\
\
           
        r_pad, valid, pad_len = self._pad_to_period(r, P)

        BN, L_pad, d = r_pad.shape
        M = L_pad // P

        windows = r_pad.reshape(BN, M, P, d)
        valid_windows = valid.reshape(BN, M, P)

        return windows, valid_windows, pad_len

    def _reverse_windows(
        self,
        windows: torch.Tensor,
        original_len: int,
    ) -> torch.Tensor:
\
\
\
\
\
           
        BN, M, P, d = windows.shape
        seq = windows.reshape(BN, M * P, d)
        seq = seq[:, :original_len, :]
        return seq

    def _hierarchical_window_merging(self, r: torch.Tensor) -> torch.Tensor:
\
\
\
\
\
           
        BN, L, d = r.shape
        mf = self.merge_factor

        pad_len = (mf - L % mf) % mf
        if pad_len > 0:
            r = F.pad(r, (0, 0, 0, pad_len))

        L_pad = r.size(1)
        L_new = L_pad // mf

        r = r.reshape(BN, L_new, mf * d)
        z = self.merge_proj(r)
        z = self.out_norm(z)

        return z

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ):
\
\
\
\
\
\
\
\
\
\
\
           
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        B, T, N, C = x.shape

        if C != self.in_channels:
            raise ValueError(
                                                                          
            )

                                                                      
                                               
                                                                      
                                                     
        xs = x.permute(0, 2, 1, 3).contiguous()
        xs = xs.reshape(B * N, T, C)

                                                                      
                          
                                                                      
        x_hat, mu, sigma = self._instance_norm(xs)

                                                                      
                            
                                                                      
        patches = self._patchify(x_hat)
                               

        e = self.patch_proj(patches)
                     

        L = e.size(1)

                                                                      
                                       
                                                                      
        r = self._light_stabilization_filter(e)
                     

                                                                      
                                      
                                                                      
        P = self._select_period(r)

                                                                      
                                  
                                                                      
        windows, valid_windows, _ = self._partition_windows(r, P)
                                 

        out_windows = self.window_attn(
            windows,
            valid_mask=valid_windows,
        )

        out_seq = self._reverse_windows(out_windows, original_len=L)
                     

        r1 = self.norm1(r + self.drop(out_seq))
                     

                                                                      
                           
                                                                      
        r1_pad, valid_pad, _ = self._pad_to_period(r1, P)
        shift_size = P // 2

        if shift_size > 0:
            r1_shift = torch.roll(r1_pad, shifts=-shift_size, dims=1)
            valid_shift = torch.roll(valid_pad, shifts=-shift_size, dims=1)
        else:
            r1_shift = r1_pad
            valid_shift = valid_pad

        BN, L_pad, d = r1_shift.shape
        M = L_pad // P

        shifted_windows = r1_shift.reshape(BN, M, P, d)
        shifted_valid_windows = valid_shift.reshape(BN, M, P)

                                                                      
                                          
                                                                      
        shifted_out_windows = self.shifted_window_attn(
            shifted_windows,
            valid_mask=shifted_valid_windows,
        )

        shifted_out = shifted_out_windows.reshape(BN, M * P, d)

                    
        if shift_size > 0:
            shifted_out = torch.roll(shifted_out, shifts=shift_size, dims=1)

        shifted_out = shifted_out[:, :L, :]

        r2 = self.norm2(r1 + self.drop(shifted_out))
                     

                                                                      
                                        
                                                                      
        zs = self._hierarchical_window_merging(r2)
                       

        L_prime = zs.size(1)
        d_prime = zs.size(2)

                                                                      
                                      
                                                                      
        z = zs.reshape(B, N, L_prime, d_prime)
                        

        if return_aux:
            aux = {
                "patch_length": self.patch_len,
                "patch_stride": self.stride,
                "patch_token_length_L": L,
                "selected_period_P": P,
                "output_length_L_prime": L_prime,
                "output_dim_d_prime": d_prime,
                "input_mu": mu,
                "input_sigma": sigma,
            }
            return z, aux

        return z
