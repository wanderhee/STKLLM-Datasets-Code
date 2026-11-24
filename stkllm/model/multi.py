\
\
\
\
   

import torch
import torch.nn as nn
import torch.nn.functional as F

def perturb_summaries(summaries, delta=0.1, sigma=0.01):
                                                                                       
    if not summaries:
        return []
    perturbed = []
    delta = float(delta)
    sigma = float(sigma)
    for item in summaries:
        ref = item["temporal"]
        rho = torch.empty((), device=ref.device, dtype=ref.dtype).uniform_(1.0 - delta, 1.0)
        noisy = {}
        for key, value in item.items():
            noisy[key] = rho * value + torch.randn_like(value) * sigma
        perturbed.append(noisy)
    return perturbed

def mode_objective(pred_loss, cons_loss, log_s, consistency_weight):
                                                 
    weight = torch.exp(-log_s)
    total = weight * (pred_loss + consistency_weight * cons_loss) + log_s
    return total.squeeze()

class MultiModeSTKLLM(nn.Module):
                                                                         

    def __init__(self, primary, primary_name, towers, consistency_weight=0.1, perturb_delta=0.1, perturb_sigma=0.01):
        super(MultiModeSTKLLM, self).__init__()
        if primary_name not in towers or towers[primary_name] is not primary:
            raise ValueError("primary encoder must be stored under primary_name")
        self.primary_name = primary_name
        self.towers = nn.ModuleDict(towers)
        self.consistency_weight = float(consistency_weight)
        self.perturb_delta = float(perturb_delta)
        self.perturb_sigma = float(perturb_sigma)
        device = primary.compute_device
        self.log_s = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, device=device))
            for name in towers
        })
        self.extra_node_logits = nn.ParameterDict()
        self.pred_heads = nn.ModuleDict()
        for name, tower in towers.items():
            if name == primary_name:
                continue
            num_nodes = int(tower.b1.shape[2])
            self.extra_node_logits[name] = nn.Parameter(torch.zeros(num_nodes, device=device))
            self.pred_heads[name] = self._new_pred_head(tower)

    @property
    def primary(self):
        return self.towers[self.primary_name]

    @property
    def kan(self):
        return self.primary.kan_branch

    @property
    def llama(self):
        return self.primary.llama3_lora

    def _new_pred_head(self, tower):
                                                                                         
        block = self.llama
        hidden = int(block.hidden_size)
        horizon = int(tower.num_for_predict)
        if tower.out_head in ("mlp", "residual"):
            mid = max(hidden // 4, 1)
            head = nn.Sequential(
                nn.Linear(hidden, mid),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mid, horizon),
            )
        else:
            head = nn.Linear(hidden, horizon)
        ref = next(block.out_layer.parameters())
        return head.to(device=ref.device, dtype=ref.dtype)

    def pred_head(self, name):
        if name == self.primary_name:
            return self.llama.out_layer
        return self.pred_heads[name]

    def node_logit(self, name):
        if name == self.primary_name:
            return self.kan.node_logit
        return self.extra_node_logits[name]

    def _others(self, name, summaries):
        return [summaries[other] for other in self.towers if other != name]

    @torch.no_grad()
    def collect_summaries(self, inputs_by_name):
                                                                                        
        summaries = {}
        for name, inputs in inputs_by_name.items():
            _, h_t, h_s = self.towers[name].encode_features(inputs)
            raw = self.kan.export_summary(h_t, h_s, node_logit=self.node_logit(name))
            summaries[name] = {key: value.detach() for key, value in raw.items()}
        return summaries

    def _project(self, tokens):
        block = self.llama
        projected = block.input_proj(tokens.to(block.llama_input_device))
        return projected

    def _predict(self, name, tower, inputs, tokens, event_prompts):
        hidden = self.llama(tokens, event_prompts=event_prompts, return_hidden=True)
        pred = self.pred_head(name)(hidden).to(self.llama.compute_device)
        if tower.out_head == "residual":
            base = inputs[:, :, 0, -1:].expand(-1, -1, tower.num_for_predict)
            return base + pred
        if tower.out_head == "mlp":
            return pred
        return F.relu(pred)

    def forward_mode(self, name, inputs, event_prompts=None, summaries=None, apply_consistency=False):
\
\
\
           
        tower = self.towers[name]
        x_in, h_t, h_s = tower.encode_features(inputs)
        others = [] if summaries is None else self._others(name, summaries)
        tokens = x_in + self.kan(h_t, h_s, others if others else None)
        pred = self._predict(name, tower, inputs, tokens, event_prompts)

        cons = pred.new_zeros(())
        if apply_consistency and others:
            perturbed = perturb_summaries(others, self.perturb_delta, self.perturb_sigma)
            tokens_pert = x_in + self.kan(h_t, h_s, perturbed)
            clean = self._project(tokens)
            shifted = self._project(tokens_pert)
            cons = (shifted - clean.detach()).abs().mean()
        return pred, cons

    def objective(self, name, pred_loss, cons_loss):
        return mode_objective(
            pred_loss, cons_loss, self.log_s[name], self.consistency_weight
        )
