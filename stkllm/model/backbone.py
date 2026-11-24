import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
                                                         

from transformers import LlamaModel, LlamaConfig, AutoTokenizer
                                                   
                              
from peft import prepare_model_for_kbit_training, get_peft_config, get_peft_model, LoraConfig, TaskType

from stkllm.model.pfs_encoder import PFSAttnEncoder
import stkllm.paths as llama_paths

class TemporalFeatureAlign(nn.Module):
                                                                           

    def __init__(self, token_len, num_timesteps, num_channels):
        super(TemporalFeatureAlign, self).__init__()
        self.proj = nn.Linear(token_len, num_timesteps)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            init.zeros_(self.proj.bias)

    def forward(self, z):
                                      
        x = self.proj(z.permute(0, 1, 3, 2))
        return x.permute(0, 2, 1, 3).contiguous()

class TimeEmb(nn.Module):

    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, dropout = 0.2):
        super(TimeEmb, self).__init__()
        self.gru = nn.GRU(input_size=num_of_vertices, hidden_size=num_of_vertices, num_layers=2, device=DEVICE)
        self.Wq1 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_timesteps).to(DEVICE))      
        self.Wk1 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_timesteps).to(DEVICE))      
        self.Wv1 = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))      
        self.d1 = num_of_timesteps

                                                                                                            
                                                                                                                

        self.dropout = nn.Dropout(dropout)
        self.FC = nn.Linear(num_of_timesteps, num_of_timesteps)
        self.ReLU = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
                      
        init.kaiming_uniform_(self.Wq1)
        init.kaiming_uniform_(self.Wk1)
        init.kaiming_uniform_(self.Wv1)

        init.kaiming_uniform_(self.FC.weight)
        if self.FC.bias is not None:
            init.zeros_(self.FC.bias)

    def forward(self, x):
\
\
\
           

        x = x[:, :, -1, :]           
        x_v = x.permute(0, 2, 1)           
        gru_in = x.permute(2, 0, 1)           
        gru_out, _ = self.gru(gru_in)                      
        gru_out = gru_out.permute(1, 0, 2)           

        Q1 = torch.matmul(gru_out, self.Wq1)                  
        K1 = torch.matmul(gru_out, self.Wk1).permute(0, 2, 1)                      
        V1 = torch.matmul(self.Wv1, x_v)                  

        product = torch.matmul(Q1, K1)                   
        T_Att = torch.div(product, math.sqrt(self.d1))              

                                                                                                

        T_Att = F.softmax(T_Att, dim=1)
        T_Att = self.dropout(T_Att)

        out = torch.matmul(T_Att, V1)                   
        out = out.permute(0, 2, 1)       

             
        out = self.FC(out)
        out = self.ReLU(out)

        out = torch.unsqueeze(out, 2)             

        return out

class Spatial_Attention(nn.Module):

    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, dropout = 0.2):
        super(Spatial_Attention, self).__init__()
        self.Wq2 = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_vertices).to(DEVICE))      
        self.Wk2 = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_vertices).to(DEVICE))      
        self.d2 = num_of_vertices
                                                                                                              
                                                                                                          
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.Wq2)
        init.kaiming_uniform_(self.Wk2)

                                       
                                       

    def forward(self, x):
\
\
\
           

        x_in = x[:, :, -1, :]           
        Q2 = torch.matmul(x_in, self.Wq2)                  
        K2 = torch.matmul(x_in, self.Wk2).permute(0, 2, 1)                      

        product = torch.matmul(Q2, K2)                   
                                                                                        

        S_Att = torch.div(product, math.sqrt(self.d2))
                                                                                
        S_Att = F.softmax(S_Att, dim=1)

        S_Att = self.dropout(S_Att)       

        return S_Att

class Graph_Conv_with_SAtt(nn.Module):

    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, adj, in_channels, out_channels, bias=True):
        super(Graph_Conv_with_SAtt, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps
        self.adj = adj.to(DEVICE)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(DEVICE))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).to(DEVICE))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, S_Att):
\
\
\
           

        outputs = []

        for time_step in range(self.num_of_timesteps):

            graph_signal = x[:, :, :, time_step]                

            L_with_SA = torch.matmul(self.adj, S_Att)                       

            support = torch.matmul(graph_signal, self.weight)                                     

            output = torch.matmul(L_with_SA.permute(0, 2, 1), support)                                                 

            if self.bias is not None:
                output = output + self.bias

            outputs.append(output.unsqueeze(-1))               

        return F.relu(torch.cat(outputs, dim=-1))               

def _attend_query_chunks(query_states, key_states, value_states, attention_mask, chunk_size, dropout_p, training):
\
\
\
\
\
       
    query_len = query_states.shape[2]
    key_len = key_states.shape[2]
    scale = 1.0 / math.sqrt(query_states.shape[-1])
    outputs = []
    for start in range(0, query_len, chunk_size):
        end = min(start + chunk_size, query_len)
        scores = torch.matmul(query_states[:, :, start:end, :], key_states.transpose(2, 3)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, start:end, :key_len]
        probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        probs = nn.functional.dropout(probs, p=dropout_p, training=training)
        outputs.append(torch.matmul(probs, value_states))
    return torch.cat(outputs, dim=2)

def _chunked_llama_attention_forward(
    attn,
    chunk_size,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    if output_attentions:
        raise RuntimeError("llama_attn_chunk_size does not support output_attentions=True")
    if getattr(attn.config, "pretraining_tp", 1) > 1:
        raise RuntimeError("llama_attn_chunk_size does not support pretraining_tp > 1")

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()
    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = attn.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, attn.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, attn.num_key_value_groups)
    value_states = repeat_kv(value_states, attn.num_key_value_groups)
    attn_output = _attend_query_chunks(
        query_states,
        key_states,
        value_states,
        attention_mask,
        chunk_size,
        attn.attention_dropout,
        attn.training,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = attn.o_proj(attn_output)
    return attn_output, None, past_key_value

def _patch_llama_attention_chunks(model, chunk_size):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = getattr(base, "layers", None)
    if layers is None and hasattr(base, "model"):
        layers = base.model.layers
    if layers is None:
        raise RuntimeError("Could not find Llama decoder layers to patch attention")
    patched = 0
    for layer in layers:
        attn = layer.self_attn
        attn.forward = (
            lambda hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,
            _attn=attn, _chunk=chunk_size, **kwargs:
            _chunked_llama_attention_forward(
                _attn,
                _chunk,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        )
        patched += 1
    return patched

class LLaMA3_Lora_block_no_emb(nn.Module):

    def __init__(
        self,
        DEVICE,
        num_for_predict,
        d_model,
        lora_layers=0,
        llama_device_map="auto",
        use_event_prompt=False,
        event_prompt_max_length=64,
        use_napg=False,
        napg_attn_heads=4,
        napg_scale_init=3.0,
        out_head="linear",
        llama_gradient_checkpointing=False,
        llama_attn_chunk_size=None,
        use_qlora=False,
    ):
        super(LLaMA3_Lora_block_no_emb, self).__init__()
        self.compute_device = DEVICE
        self.use_qlora = bool(use_qlora)
        self.use_event_prompt = bool(use_event_prompt)
        self.event_prompt_max_length = int(event_prompt_max_length)
        self.use_napg = bool(use_napg)
        self.llama_gradient_checkpointing = bool(llama_gradient_checkpointing)
        if llama_attn_chunk_size is None:
                                                                                    
                                                                               
                                       
            self.llama_attn_chunk_size = 256 if self.llama_gradient_checkpointing else 0
        else:
            self.llama_attn_chunk_size = int(llama_attn_chunk_size)
        if self.llama_attn_chunk_size < 0:
            raise ValueError("llama_attn_chunk_size must be >= 0, got %d" % self.llama_attn_chunk_size)
        if self.llama_attn_chunk_size > 0 and not self.llama_gradient_checkpointing:
            self.llama_gradient_checkpointing = True
        self.out_head = str(out_head or "linear").strip().lower()
        if self.out_head not in ("linear", "mlp", "residual"):
            raise ValueError("out_head must be one of: linear, mlp, residual; got %r" % out_head)
        self.num_for_predict = int(num_for_predict)
                           
        self.pretrained_model_path = llama_paths.LLAMA_MODEL_PATH
        if not os.path.isabs(self.pretrained_model_path):
            raise ValueError("LLaMA 权重路径必须是绝对路径: %s" % self.pretrained_model_path)
        if not os.path.isdir(self.pretrained_model_path) or not os.path.exists(os.path.join(self.pretrained_model_path, "config.json")):
            raise FileNotFoundError("LLaMA 权重目录不存在: %s" % self.pretrained_model_path)

                    
        config = LlamaConfig.from_pretrained(self.pretrained_model_path)
        if self.use_napg and self.use_event_prompt:
                                                                                                    
            config._attn_implementation = "eager"
        if self.use_qlora:
            try:
                import bitsandbytes              
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                                                                                       
                ) from exc
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
                                                                              
                                                                
            import transformers.modeling_utils as _tf_modeling

            _dispatch_model = _tf_modeling.dispatch_model

            def _dispatch_quantized(model, *args, **kwargs):
                kwargs["force_hooks"] = True
                return _dispatch_model(model, *args, **kwargs)

            _tf_modeling.dispatch_model = _dispatch_quantized
            try:
                self.llama3 = LlamaModel.from_pretrained(
                    self.pretrained_model_path,
                    config=config,
                    quantization_config=quant_config,
                    device_map=llama_device_map,
                )
            finally:
                _tf_modeling.dispatch_model = _dispatch_model
        else:
            self.llama3 = LlamaModel.from_pretrained(
                self.pretrained_model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map=llama_device_map,
            )
        
        num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
        self.lora_layers = int(lora_layers) if lora_layers is not None else 0
        self.lora_layers = max(0, self.lora_layers)

        if self.lora_layers > 0 and num_hidden_layers > 0:
            limited_layers = min(self.lora_layers, num_hidden_layers)
            target_modules = []
            for layer_id in range(limited_layers):
                target_modules.append(f"layers.{layer_id}.self_attn.q_proj")
                target_modules.append(f"layers.{layer_id}.self_attn.v_proj")
        else:
            target_modules = ["q_proj", "v_proj"]

                  
        self.peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=16, 
            lora_dropout=0.1,                      
            target_modules=target_modules,              
            bias="none"
        )
        
                        
                                                                                  
                                               
        if self.use_qlora:
                                                                              
                                                                 
            self.llama3 = prepare_model_for_kbit_training(
                self.llama3, use_gradient_checkpointing=False
            )
        else:
            self.llama3 = prepare_model_for_kbit_training(self.llama3)
        if self.llama_gradient_checkpointing:
            self.llama3.config.use_cache = False
            self.llama3.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.lora_llama3 = get_peft_model(self.llama3, self.peft_config)
        if self.use_qlora:
                                                                         
                                                                    
            self.qlora_activation_dtype = self._get_embed_tokens().weight.dtype
        if self.llama_attn_chunk_size > 0:
            _patch_llama_attention_chunks(self.lora_llama3, self.llama_attn_chunk_size)
        
        self.input_dim = d_model
        self.hidden_size = config.hidden_size
        if self.input_dim != self.hidden_size:
            self.input_proj = nn.Linear(self.input_dim, self.hidden_size)
        else:
            self.input_proj = nn.Identity()

        if self.out_head in ("mlp", "residual"):
            mid = max(self.hidden_size // 4, 1)
            self.out_layer = nn.Sequential(
                nn.Linear(self.hidden_size, mid),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mid, self.num_for_predict),
            )
        else:
            self.out_layer = nn.Linear(self.hidden_size, self.num_for_predict)

                                                    
        self.llama_input_device = self._resolve_llama_input_device()
        self.llama_output_device = self._resolve_llama_output_device()
        self.input_proj.to(self.llama_input_device)
        self.out_layer.to(self.llama_output_device)

        self.tokenizer = None
        if self.use_event_prompt:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.napg_gate = None
        if self.use_napg:
            if not self.use_event_prompt:
                self.use_napg = False
            else:
                gate_hidden = max(self.hidden_size // 4, 1)
                self.napg_gate = nn.Sequential(
                    nn.Linear(self.hidden_size, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, 1),
                )
                self._init_napg_gate()
                self.napg_scale = nn.Parameter(
                    torch.tensor(float(napg_scale_init), dtype=torch.float32)
                )
                self.napg_gate.to(self.llama_input_device)
                self.napg_scale.data = self.napg_scale.data.to(self.llama_input_device)

    def _init_napg_gate(self):
\
\
\
           
        last = self.napg_gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _get_embed_tokens(self):
        return self.lora_llama3.get_input_embeddings()

    def _encode_event_prompts(self, event_prompts):
        if not self.use_event_prompt or event_prompts is None:
            return None, None, 0

        prompt_texts = ["" if p is None else str(p) for p in event_prompts]
        if not any(text.strip() for text in prompt_texts):
            return None, None, 0

        encoded = self.tokenizer(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=self.event_prompt_max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.llama_input_device)
        attention_mask = encoded["attention_mask"].to(self.llama_input_device)

        embed_tokens = self._get_embed_tokens()
        model_dtype = self._model_dtype()
        prompt_embeds = embed_tokens(input_ids).to(dtype=model_dtype, device=self.llama_input_device)
        return prompt_embeds, attention_mask, prompt_embeds.shape[1]

    def _compute_napg_log_bias(self, node_feats):
\
\
\
\
\
           
        raw = self.napg_gate(node_feats)
        scale = torch.clamp(self.napg_scale, min=1e-3)
        scaled = scale * torch.tanh(raw)
        return torch.minimum(scaled, torch.zeros_like(scaled))

    def _build_napg_attention_bias(
        self,
        log_gate_bias,
        prompt_len,
        num_nodes,
        prompt_attention_mask,
        dtype,
        device,
    ):
\
\
\
\
\
           
        batch_size = log_gate_bias.shape[0]
        seq_len = prompt_len + num_nodes
        min_value = torch.finfo(dtype).min

        bias = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        bias = bias.masked_fill(causal.view(1, 1, seq_len, seq_len), min_value)

        if prompt_attention_mask is not None:
            pad_keys = prompt_attention_mask == 0
            if pad_keys.any():
                bias[:, 0, :, :prompt_len] = bias[:, 0, :, :prompt_len].masked_fill(
                    pad_keys.unsqueeze(1),
                    min_value,
                )

        node_prompt_bias = log_gate_bias.squeeze(-1).unsqueeze(-1).expand(-1, -1, prompt_len)
        valid_prompt = prompt_attention_mask.unsqueeze(1).expand(-1, num_nodes, -1)
        node_prompt_bias = node_prompt_bias.masked_fill(valid_prompt == 0, 0.0)
        bias[:, 0, prompt_len:prompt_len + num_nodes, :prompt_len] += node_prompt_bias
        return bias

    def _resolve_llama_input_device(self):
        try:
            return self.lora_llama3.base_model.model.embed_tokens.weight.device
        except Exception:
            return next(self.lora_llama3.parameters()).device

    def _resolve_llama_output_device(self):
        try:
            return self.lora_llama3.base_model.model.norm.weight.device
        except Exception:
            return next(self.lora_llama3.parameters()).device

    def _align_prompt_and_nodes(self, prompt_embeds, node_feats, prompt_attention_mask):
                                                                        
        return prompt_embeds, node_feats

    def _model_dtype(self):
        if self.use_qlora:
            return self.qlora_activation_dtype
        return next(self.lora_llama3.parameters()).dtype

    def forward(self, x, event_prompts=None, return_hidden=False):
\
\
\
\
\
           
        x = x.to(self.llama_input_device)
        x = self.input_proj(x)
        model_dtype = self._model_dtype()
        x = x.to(dtype=model_dtype, device=self.llama_input_device)

        prompt_embeds, prompt_attention_mask, prompt_len = self._encode_event_prompts(event_prompts)
        if prompt_embeds is not None and prompt_len > 0:
            batch_size, num_nodes, hidden_size = x.shape
            prompt_embeds, x = self._align_prompt_and_nodes(
                prompt_embeds, x, prompt_attention_mask
            )
            inputs_embeds = torch.cat([prompt_embeds, x], dim=1)
            if self.use_napg:
                log_gate_bias = self._compute_napg_log_bias(x)
                attention_mask = self._build_napg_attention_bias(
                    log_gate_bias,
                    prompt_len,
                    num_nodes,
                    prompt_attention_mask,
                    dtype=model_dtype,
                    device=self.llama_input_device,
                )
            else:
                node_attention = torch.ones(
                    batch_size,
                    num_nodes,
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                )
                attention_mask = torch.cat([prompt_attention_mask, node_attention], dim=1)
            outputs = self.lora_llama3(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state[:, prompt_len:, :]
        else:
            outputs = self.lora_llama3(inputs_embeds=x, return_dict=True)
            hidden_states = outputs.last_hidden_state

        hidden_states = hidden_states.to(self.llama_output_device)
        if return_hidden:
            return hidden_states
        output = self.out_layer(hidden_states)             
        return output.to(self.compute_device)

class STATPLLM(nn.Module):

    def __init__(
        self,
        DEVICE,
        num_of_vertices,
        num_of_timesteps,
        adj,
        in_channels,
        nb_filter,
        time_strides,
        d_model,
        num_for_predict,
        lora_layers=0,
        llama_device_map="auto",
        temporal_encoder_cfg=None,
        use_event_prompt=False,
        event_prompt_max_length=64,
        use_napg=False,
        napg_attn_heads=4,
        napg_scale_init=3.0,
        use_temporal_encoder=True,
        use_spatial_encoder=True,
        use_st_gated_residual=False,
        out_head="linear",
        llama_gradient_checkpointing=False,
        llama_attn_chunk_size=None,
        use_qlora=False,
        skip_llama=False,
    ):
        super(STATPLLM, self).__init__()
        self.skip_llama = bool(skip_llama)
        self.compute_device = DEVICE
        self.use_temporal_encoder = bool(use_temporal_encoder)
        self.use_spatial_encoder = bool(use_spatial_encoder)
        self.use_st_gated_residual = bool(use_st_gated_residual)
        self.out_head = str(out_head or "linear").strip().lower()
        self.num_for_predict = int(num_for_predict)
        temporal_encoder_cfg = temporal_encoder_cfg or {"type": "gru"}
        self.temporal_encoder_type = temporal_encoder_cfg.get("type", "gru")

        self.TE = None
        self.pfs_encoder = None
        self.temporal_align = None
        self.pfs_channel_proj = None

        self.b1 = nn.Parameter(
            torch.FloatTensor(1, nb_filter, num_of_vertices, num_of_timesteps).to(DEVICE)
        )

        if self.use_temporal_encoder and self.temporal_encoder_type == "pfs":
            pfs_kwargs = dict(temporal_encoder_cfg["pfs"])
            pfs_out_dim = int(pfs_kwargs.get("out_dim", nb_filter))
                                     
            pfs_kwargs["in_channels"] = nb_filter
            self.pfs_encoder = PFSAttnEncoder(**pfs_kwargs)
            self.temporal_align = TemporalFeatureAlign(
                token_len=temporal_encoder_cfg["pfs_token_len"],
                num_timesteps=num_of_timesteps,
                num_channels=pfs_out_dim,
            )
            if pfs_out_dim != nb_filter:
                self.pfs_channel_proj = nn.Conv2d(
                    pfs_out_dim, nb_filter, kernel_size=(1, 1)
                )
            te_conv_in = in_channels
        elif self.use_temporal_encoder:
            self.TE = TimeEmb(DEVICE, num_of_vertices, num_of_timesteps)
            te_conv_in = in_channels
        else:
            te_conv_in = in_channels

        self.TE_conv = nn.Conv2d(
            te_conv_in,
            nb_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )

        self.SA = None
        self.SE = None
        self.Wv2 = None
        self.b2 = None
        if self.use_spatial_encoder:
            self.SA = Spatial_Attention(DEVICE, num_of_vertices, num_of_timesteps)
            self.SE = Graph_Conv_with_SAtt(
                DEVICE,
                num_of_vertices,
                num_of_timesteps,
                adj,
                in_channels=in_channels,
                out_channels=nb_filter,
            )
            self.Wv2 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))
            self.b2 = nn.Parameter(
                torch.FloatTensor(1, nb_filter, num_of_vertices, num_of_timesteps).to(DEVICE)
            )
        self.residual_conv = nn.Conv2d(in_channels, nb_filter, kernel_size=(1, 1), stride=(1, time_strides))
                                                                                    
        self.st_fuse_gate = None
        if self.use_st_gated_residual:
            self.st_fuse_gate = nn.Parameter(
                torch.zeros(1, nb_filter, 1, 1, device=DEVICE)
            )
        self.GELU = nn.GELU()
        self.leaky_relu = nn.LeakyReLU()
        self.ln = nn.LayerNorm(nb_filter)                       
        self.input_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, 1))
              
        self.llama3_lora = None
        if self.skip_llama:
            self.reset_parameters()
            self._move_encoder(DEVICE)
            return
        self.llama3_lora = LLaMA3_Lora_block_no_emb(
            DEVICE,
            num_for_predict,
            d_model,
            lora_layers=lora_layers,
            llama_device_map=llama_device_map,
            use_event_prompt=use_event_prompt,
            event_prompt_max_length=event_prompt_max_length,
            use_napg=use_napg,
            napg_attn_heads=napg_attn_heads,
            napg_scale_init=napg_scale_init,
            out_head=self.out_head,
            llama_gradient_checkpointing=llama_gradient_checkpointing,
            llama_attn_chunk_size=llama_attn_chunk_size,
            use_qlora=use_qlora,
        )
        self.reset_parameters()

                                                          
        self._move_encoder(DEVICE)

    def _move_encoder(self, device):
        layers_to_device = [
            self.residual_conv,
            self.GELU,
            self.leaky_relu,
            self.ln,
            self.input_conv,
            self.TE_conv,
        ]
        if self.SA is not None:
            layers_to_device.append(self.SA)
        if self.SE is not None:
            layers_to_device.append(self.SE)
        if self.TE is not None:
            layers_to_device.append(self.TE)
        if self.pfs_encoder is not None:
            layers_to_device.extend([self.pfs_encoder, self.temporal_align])
            if self.pfs_channel_proj is not None:
                layers_to_device.append(self.pfs_channel_proj)
        for layer in layers_to_device:
            layer.to(device)

    def reset_parameters(self):
        if self.Wv2 is not None:
            init.kaiming_uniform_(self.Wv2)
        if self.b2 is not None:
            init.kaiming_uniform_(self.b2)
        init.kaiming_uniform_(self.b1)

    def extract_temporal_features(self, x):
        if not self.use_temporal_encoder:
            x_conv = self.TE_conv(x.permute(0, 2, 1, 3))
            return F.relu(x_conv + self.b1)

        if self.temporal_encoder_type == "pfs":
                                               
            x_conv = self.TE_conv(x.permute(0, 2, 1, 3))
            x_skip = x_conv

            z = self.pfs_encoder(x_conv.permute(0, 3, 2, 1).contiguous())
            x_pfs = self.temporal_align(z)
            if self.pfs_channel_proj is not None:
                x_pfs = self.pfs_channel_proj(x_pfs)

            return F.relu(x_pfs + x_skip + self.b1)

        x_te = self.TE(x).permute(0, 2, 1, 3)
        return F.relu(self.TE_conv(x_te) + self.b1)

    def _kan_residual(self, x_in, h_t, h_s, other_modes=None):
                                                                      
        return x_in

    def encode_features(self, x):
\
\
\
           
        x_TE = self.extract_temporal_features(x)

        if self.use_spatial_encoder:
            SA = self.SA(x)
            x_s = torch.matmul(self.Wv2, x.permute(0, 2, 1, 3))
            x_s = x_s.permute(0, 2, 1, 3)
            x_SE = self.SE(x_s, SA)
            x_SE = F.relu(x_SE.permute(0, 2, 1, 3) + self.b2)
        else:
            x_SE = torch.zeros_like(x_TE)

                                        
        x_st = self.GELU(x_TE + x_SE)

        if self.use_st_gated_residual:
                                                                       
            x_main = self.residual_conv(x.permute(0, 2, 1, 3))
            gate = torch.sigmoid(self.st_fuse_gate)
            x_fused = x_main + gate * x_st
        else:
                                                              
            x_fused = x_st

        x_in = x_fused.permute(0, 3, 2, 1)                      
        x_in = self.ln(x_in).permute(0, 2, 3, 1)                      
        x_in = self.input_conv(x_in.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)                  
        h_t = x_TE.permute(0, 2, 3, 1).contiguous()              
        h_s = x_SE.permute(0, 2, 3, 1).contiguous()
        return x_in, h_t, h_s

    def forward(self, x, event_prompts=None, other_modes=None):
\
\
\
\
\
           
        if self.llama3_lora is None:
            raise RuntimeError("This encoder has no LLaMA block. Use the shared one.")

        x_in, h_t, h_s = self.encode_features(x)
        x_in = self._kan_residual(x_in, h_t, h_s, other_modes)
        pred = self.llama3_lora(x_in, event_prompts=event_prompts)         

        if self.out_head == "residual":
                                                                    
            y_base = x[:, :, 0, -1:].expand(-1, -1, self.num_for_predict)
            return y_base + pred
        if self.out_head == "mlp":
            return pred
                                                             
        return F.relu(pred)

def make_STATPLLM_model(
    DEVICE,
    len_input,
    time_strides,
    in_channels,
    nb_chev_filter,
    adj_mx,
    num_of_vertices,
    d_model,
    num_for_predict,
    lora_layers=0,
    llama_device_map="auto",
    temporal_encoder_cfg=None,
    use_event_prompt=False,
    event_prompt_max_length=64,
    use_napg=False,
    napg_attn_heads=4,
    napg_scale_init=3.0,
    use_temporal_encoder=True,
    use_spatial_encoder=True,
    use_st_gated_residual=False,
    out_head="linear",
    llama_gradient_checkpointing=False,
    llama_attn_chunk_size=None,
    use_qlora=False,
    skip_llama=False,
):

    model = STATPLLM(
        DEVICE=DEVICE,
        num_of_vertices=num_of_vertices,
        num_of_timesteps=len_input,
        adj=adj_mx,
        in_channels=in_channels,
        nb_filter=nb_chev_filter,
        time_strides=time_strides,
        d_model=d_model,
        num_for_predict=num_for_predict,
        lora_layers=lora_layers,
        llama_device_map=llama_device_map,
        temporal_encoder_cfg=temporal_encoder_cfg,
        use_event_prompt=use_event_prompt,
        event_prompt_max_length=event_prompt_max_length,
        use_napg=use_napg,
        napg_attn_heads=napg_attn_heads,
        napg_scale_init=napg_scale_init,
        use_temporal_encoder=use_temporal_encoder,
        use_spatial_encoder=use_spatial_encoder,
        use_st_gated_residual=use_st_gated_residual,
        out_head=out_head,
        llama_gradient_checkpointing=llama_gradient_checkpointing,
        llama_attn_chunk_size=llama_attn_chunk_size,
        use_qlora=use_qlora,
        skip_llama=skip_llama,
    )

    return model
