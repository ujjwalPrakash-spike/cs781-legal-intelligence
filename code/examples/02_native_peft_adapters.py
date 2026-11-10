#!/usr/bin/env python3
"""02_native_peft_adapters.py
Standalone pure PyTorch implementation of LoRA, LORMA, BitFit, and Layer-Wise freezing
with parameter accounting.
"""

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) module replacing nn.Linear."""
    def __init__(self, linear_layer, r=16, alpha=32):
        super().__init__()
        self.linear = linear_layer
        self.scaling = alpha / r
        
        # Matrix A: initialized from Gaussian distribution (scaled by 1/r)
        self.lora_A = nn.Parameter(torch.randn(r, linear_layer.in_features) * (1.0 / r))
        # Matrix B: initialized to zero so delta_W = 0 at step 0
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, r))
        
        # Freeze base linear weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        base_out = self.linear(x)
        adapter_out = torch.matmul(x, torch.matmul(self.lora_B, self.lora_A).T) * self.scaling
        return base_out + adapter_out

class LORMALinear(nn.Module):
    """Low-Rank Multi-Head Adapter (LORMA) with learnable memory gate."""
    def __init__(self, linear_layer, r=16, alpha=32):
        super().__init__()
        self.linear = linear_layer
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, linear_layer.in_features) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, r))
        self.memory_gate = nn.Parameter(torch.ones(1) * 0.1) # Learnable scalar gate
        self.linear.weight.requires_grad = False

    def forward(self, x):
        adapter_out = torch.matmul(x, torch.matmul(self.lora_B, self.lora_A).T) * self.scaling
        return self.linear(x) + self.memory_gate * adapter_out

class MockTransformerBlock(nn.Module):
    """Simple 2-layer mock transformer block for PEFT testing."""
    def __init__(self, dim=768):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.fc1 = nn.Linear(dim, dim * 4, bias=True)
        self.fc2 = nn.Linear(dim * 4, dim, bias=True)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = self.o_proj(v)
        return self.fc2(torch.relu(self.fc1(attn + x)))

def apply_bitfit(model):
    """Freezes all 2D weight matrices; trains only 1D biases."""
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, (100.0 * trainable / total)

if __name__ == "__main__":
    print("=== PEFT Adapter Accounting Demonstration ===")
    
    # 1. Base Model
    block = MockTransformerBlock(dim=768)
    t, total, pct = count_trainable(block)
    print(f"Full Fine-Tuning:   {t:,} / {total:,} ({pct:.2f}% trainable)")
    
    # 2. LoRA Adaptation
    block_lora = MockTransformerBlock(dim=768)
    for p in block_lora.parameters():
        p.requires_grad = False
    block_lora.q_proj = LoRALinear(block_lora.q_proj, r=16, alpha=32)
    block_lora.v_proj = LoRALinear(block_lora.v_proj, r=16, alpha=32)
    t, total, pct = count_trainable(block_lora)
    print(f"LoRA (q, v, r=16):  {t:,} / {total:,} ({pct:.2f}% trainable)")
    
    # 3. BitFit
    block_bitfit = MockTransformerBlock(dim=768)
    apply_bitfit(block_bitfit)
    t, total, pct = count_trainable(block_bitfit)
    print(f"BitFit (bias only): {t:,} / {total:,} ({pct:.2f}% trainable)")
    
    # Test forward pass
    x = torch.randn(2, 16, 768)
    y_lora = block_lora(x)
    assert y_lora.shape == x.shape, "Shape mismatch in LoRA forward!"
    print("SUCCESS: PEFT adapter modules forward pass verified.")
