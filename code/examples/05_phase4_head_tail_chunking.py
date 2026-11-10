#!/usr/bin/env python3
"""
05_phase4_head_tail_chunking.py
--------------------------------
CS781 Capstone - Viva Voce Demonstration Script
Phase 4: Legal Bail Prediction with Head-Tail Chunking & Threshold Optimization

Why this matters for Viva:
1. Legal documents exceed the 512-token BERT context window (averaging 3,000-5,000 tokens).
2. Naive truncation loses the most critical signal: the FIR/charges are at the start (Head),
   while the judge's reasoning and bail order are at the end (Tail). Middle sections
   contain repetitive witness depositions.
3. Class imbalance (granted vs denied bail) makes standard 0.5 threshold sub-optimal.
   Calibrating tau to 0.325 significantly boosts macro F1.
"""

import torch
import torch.nn as nn
import numpy as np

class HeadTailClassifier(nn.Module):
    """
    Encodes head and tail chunks independently through a shared backbone,
    then combines their pooled representations for classification.
    """
    def __init__(self, hidden_dim=768, num_classes=1):
        super().__init__()
        # In practice this is InLegalBERT; here simulated with linear projection + LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # Concatenate pooled head [B, H] and pooled tail [B, H] -> [B, 2*H]
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, head_embeddings, tail_embeddings):
        # head_embeddings: [B, seq_len, hidden_dim]
        # tail_embeddings: [B, seq_len, hidden_dim]
        
        # Mean pooling across token length
        h_head = self.encoder(head_embeddings).mean(dim=1)  # [B, H]
        h_tail = self.encoder(tail_embeddings).mean(dim=1)  # [B, H]
        
        combined = torch.cat([h_head, h_tail], dim=-1)      # [B, 2*H]
        logits = self.classifier(combined)                   # [B, 1]
        return logits


def extract_head_tail(token_ids, max_head=256, max_tail=256):
    """
    Splits long sequence into head and tail tokens.
    """
    total_len = len(token_ids)
    if total_len <= (max_head + max_tail):
        # Fits within budget
        head = token_ids[:max_head]
        tail = token_ids[max_head:]
    else:
        head = token_ids[:max_head]
        tail = token_ids[-max_tail:]
    return head, tail


def find_optimal_threshold(y_true, y_probs):
    """
    Sweeps thresholds from 0.05 to 0.95 to find tau that maximizes binary F1 score.
    """
    best_f1 = -1.0
    best_tau = 0.5
    
    thresholds = np.linspace(0.05, 0.95, 91)
    for tau in thresholds:
        preds = (y_probs >= tau).astype(int)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            
    return best_tau, best_f1


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    print("=== Phase 4: Head-Tail Chunking & Threshold Optimization ===")
    
    # 1. Simulate document length distribution
    doc_tokens = list(range(1, 3450)) # 3,449 tokens (typical Indian High Court bail order)
    print(f"Raw legal document length: {len(doc_tokens)} tokens")
    
    head_tokens, tail_tokens = extract_head_tail(doc_tokens, max_head=256, max_tail=256)
    print(f"Extracted Head tokens: {len(head_tokens)} (tokens 1 to {head_tokens[-1]})")
    print(f"Extracted Tail tokens: {len(tail_tokens)} (tokens {tail_tokens[0]} to {tail_tokens[-1]})")
    print(f"Skipped middle tokens: {tail_tokens[0] - head_tokens[-1] - 1} repetitive testimony tokens\n")
    
    # 2. Forward pass through HeadTailClassifier
    batch_size = 8
    seq_len = 256
    hidden_dim = 768
    
    model = HeadTailClassifier(hidden_dim=hidden_dim)
    head_tensors = torch.randn(batch_size, seq_len, hidden_dim)
    tail_tensors = torch.randn(batch_size, seq_len, hidden_dim)
    
    logits = model(head_tensors, tail_tensors)
    probs = torch.sigmoid(logits).squeeze(-1)
    print(f"Classifier output logits shape: {logits.shape}")
    print(f"Batch predicted probabilities: {probs.detach().numpy().round(3)}\n")
    
    # 3. Simulate Validation Set Threshold Calibration
    n_val = 500
    # Simulate imbalanced ground truth (30% bail granted, 70% denied)
    val_labels = (np.random.rand(n_val) < 0.30).astype(int)
    # Simulated model probabilities correlated with ground truth + noise
    noise = np.random.normal(0, 0.2, n_val)
    val_probs = np.clip(val_labels * 0.5 + 0.15 + noise, 0.01, 0.99)
    
    # Default 0.5 threshold evaluation
    default_preds = (val_probs >= 0.5).astype(int)
    tp = np.sum((default_preds == 1) & (val_labels == 1))
    fp = np.sum((default_preds == 1) & (val_labels == 0))
    fn = np.sum((default_preds == 0) & (val_labels == 1))
    f1_default = 2 * tp / (2 * tp + fp + fn)
    
    best_tau, f1_best = find_optimal_threshold(val_labels, val_probs)
    
    print(f"Validation Results (N={n_val}):")
    print(f"  Default Threshold (tau=0.500): F1 = {f1_default:.4f}")
    print(f"  Optimal Threshold (tau={best_tau:.3f}): F1 = {f1_best:.4f}")
    print(f"  Delta Gain: +{(f1_best - f1_default)*100:.2f}% F1 points by threshold tuning")
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
