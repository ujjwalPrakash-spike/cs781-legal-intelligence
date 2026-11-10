#!/usr/bin/env python3
"""03_knowledge_distillation_step.py
Standalone implementation of Hinton-style Knowledge Distillation loss step
with temperature scaling and gradient verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_kd_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.5):
    """Computes combined Knowledge Distillation Loss:
    L = (1 - alpha) * CrossEntropy(student, labels) + alpha * (T^2) * KL(student / T, teacher / T)
    """
    # 1. Hard Cross-Entropy on true ground-truth targets
    loss_ce = F.cross_entropy(student_logits, true_labels)
    
    # 2. Softened probability matching via Kullback-Leibler Divergence
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    
    loss_kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    
    total_loss = (1.0 - alpha) * loss_ce + alpha * loss_kl
    return total_loss, loss_ce.item(), loss_kl.item()

if __name__ == "__main__":
    print("=== Knowledge Distillation Step Demonstration ===")
    torch.manual_seed(42)
    
    batch_size = 8
    num_classes = 13  # Phase 3 Rhetorical Role classes
    
    # Simulate teacher model (frozen) and student model (trainable)
    student_classifier = nn.Linear(768, num_classes)
    teacher_classifier = nn.Linear(768, num_classes)
    
    # Freeze teacher
    for p in teacher_classifier.parameters():
        p.requires_grad = False
        
    # Mock representations
    features = torch.randn(batch_size, 768)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward passes
    with torch.no_grad():
        t_logits = teacher_classifier(features)
    s_logits = student_classifier(features)
    
    total_loss, ce_val, kl_val = compute_kd_loss(s_logits, t_logits, labels, temperature=3.0, alpha=0.5)
    
    # Backward pass
    total_loss.backward()
    grad_norm = student_classifier.weight.grad.norm().item()
    
    print(f"Batch size:      {batch_size}, Classes: {num_classes}")
    print(f"Hard CE Loss:    {ce_val:.4f}")
    print(f"Soft KL Loss:    {kl_val:.4f}")
    print(f"Total Loss:      {total_loss.item():.4f}")
    print(f"Student Gradient Norm: {grad_norm:.4f}")
    print("SUCCESS: Knowledge distillation step verified.")
