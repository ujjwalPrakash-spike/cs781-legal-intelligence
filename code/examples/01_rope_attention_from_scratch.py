#!/usr/bin/env python3
"""01_rope_attention_from_scratch.py
Standalone pure PyTorch implementation of Rotary Positional Embeddings (RoPE)
and Scaled Dot-Product Attention from first principles.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    """Computes Rotary Position Embeddings (RoPE) as introduced by Su et al. (2021)."""
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        # theta_i = 10000^(-2(i-1)/d) for i in [1, d/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cosine and sine tables up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # Shape: [max_seq_len, dim / 2]
        emb = torch.cat((freqs, freqs), dim=-1) # Shape: [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        # Returns cos and sin slices matching current sequence length
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x):
    """Rotates half the hidden dimensions of the input vector: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies 2D complex plane rotation to Query and Key tensors.
    Shape of q, k: [Batch, NumHeads, SeqLen, HeadDim]
    Shape of cos, sin: [SeqLen, HeadDim] -> reshaped to [1, 1, SeqLen, HeadDim]
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoPEMultiHeadAttention(nn.Module):
    """12-head causal multi-head self-attention layer with RoPE."""
    def __init__(self, hidden_dim=768, num_heads=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 768 / 12 = 64
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.rope = RotaryEmbedding(dim=self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear Projections: [B, S, H] -> [B, NumHeads, S, HeadDim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Apply Rotary Position Embeddings to Q and K
        cos, sin = self.rope(v, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 3. Scaled Dot-Product Attention with Causal Mask
        # Attention scores shape: [B, NumHeads, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax over last dimension
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. Multiply by Values: [B, NumHeads, S, HeadDim]
        context = torch.matmul(attn_weights, v)
        
        # 5. Concatenate heads back to [B, S, HiddenDim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.o_proj(context)

if __name__ == "__main__":
    print("Testing RoPEMultiHeadAttention...")
    batch_size, seq_len, hidden_dim = 2, 8, 768
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)
    
    layer = RoPEMultiHeadAttention(hidden_dim=hidden_dim, num_heads=12)
    output = layer(dummy_input)
    
    print(f"Input shape:  {list(dummy_input.shape)}")
    print(f"Output shape: {list(output.shape)}")
    assert output.shape == dummy_input.shape, "Shape mismatch in output!"
    print("SUCCESS: RoPE self-attention forward pass verified.")
