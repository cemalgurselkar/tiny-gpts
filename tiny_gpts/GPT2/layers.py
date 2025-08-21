import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, n_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim) if bias else None)
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CasualSelfAttention(nn.Module):
    """ 
    Mask Multi-Head Attention
    """
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(F, 'scaled_dot_product_attention')

        if not self.flash:
            print("Warning: using slow attention. Flash attention requires PyTorch >= 2.0")

            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() #batch_size, sequence_lenght, embedding_dim

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C// self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C// self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C// self.n_head).transpose(1,2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_project = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.dropout(self.c_project(x))
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CasualSelfAttention(config)
        self.layer2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attention(self.layer1(x))
        x = x + self.mlp(self.layer2(x))
        return x