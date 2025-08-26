import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization:
        norm_a = a / RMS(a)
        output = weight * norm_a
        RMS(a) = sqrt(mean(a^2))
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class RotaryEmbedding(nn.Module):
    
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        frequences = torch.einsum("i, j->ij", t, inv_freq) #Einstein Summation
        emb = torch.cat((frequences, frequences), dim=-1)

        #self.register_buffer(name, tensor)
        #Registers a tensor as a buffer in the module, meaning: 
            #It will be moved to the GPU when the model is moved (e.g., .cuda()).
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DeepSeekAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # source: https://docs.pytorch.org/docs/stable/generated/torch.repeat_interleave.html
        k = k.repeat_interleave(self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_kv_heads, dim=1)

        attn_weights = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output)


class DeepSeekMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_porj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_porj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.self_attention = DeepSeekAttention(config)
        self.mlp = DeepSeekMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, attention_mask=None, position_ids=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attention(x, attention_mask, position_ids)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(self.mlp(x))
        x = residual + x
        return x

class DeepSeekModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        x = self.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min
        
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        
        return self.norm(x)

class DeepSeekCasualLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = DeepSeekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask)
        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}

    def generate(self, input_ids, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(input_ids).logits[:, -1, :] / temperature
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids