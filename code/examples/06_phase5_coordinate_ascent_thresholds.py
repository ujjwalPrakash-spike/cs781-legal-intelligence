#!/usr/bin/env python3
"""
06_phase5_coordinate_ascent_thresholds.py
-------------------------------------------
CS781 Capstone - Viva Voce Demonstration Script
Phase 5: Coordinate Ascent Threshold Calibration for Multi-Label Statute Identification

Why this matters for Viva:
1. Legal statute prediction has 100 classes with severe power-law class imbalance.
2. A static 0.5 threshold causes zero predictions for rare statutes (e.g. IPC Section 304B vs IPC Section 302),
   plunging Macro-F1 to ~0.18.
3. Gradient descent cannot optimize Macro-F1 directly because argmax/thresholding is a
   non-differentiable step function.
4. Coordinate ascent iteratively sweeps candidate thresholds per class while holding other
   classes fixed, directly maximizing the target evaluation metric.
"""

import numpy as np

def compute_per_class_f1(y_true, y_pred_binary):
    """
    Computes F1 for each class independently.
    y_true: [N, C] binary matrix
    y_pred_binary: [N, C] binary matrix
    """
    tp = np.sum((y_pred_binary == 1) & (y_true == 1), axis=0)
    fp = np.sum((y_pred_binary == 1) & (y_true == 0), axis=0)
    fn = np.sum((y_pred_binary == 0) & (y_true == 1), axis=0)
    
    denominator = (2 * tp + fp + fn)
    # Avoid zero division: if no true or predicted positives, F1 = 0.0
    f1s = np.zeros(y_true.shape[1], dtype=np.float32)
    valid_mask = denominator > 0
    f1s[valid_mask] = (2 * tp[valid_mask]) / denominator[valid_mask]
    return f1s


def coordinate_ascent_threshold_search(y_true, y_probs, num_steps=35, max_iters=2):
    """
    Calibrates per-class decision thresholds using coordinate ascent.
    y_true: [N, C] ground truth
    y_probs: [N, C] predicted sigmoid probabilities
    """
    num_samples, num_classes = y_true.shape
    thresholds = np.full(num_classes, 0.50, dtype=np.float32)
    candidate_tau = np.linspace(0.05, 0.90, num_steps)
    
    # Baseline with tau = 0.5
    current_preds = (y_probs >= thresholds).astype(np.int32)
    current_f1s = compute_per_class_f1(y_true, current_preds)
    current_macro_f1 = np.mean(current_f1s)
    
    print(f"Starting Coordinate Ascent: Initial Macro-F1 = {current_macro_f1:.4f}")
    
    for iteration in range(max_iters):
        improved = False
        for c in range(num_classes):
            best_tau_c = thresholds[c]
            best_f1_c = current_f1s[c]
            
            # Save original prediction column
            original_col = current_preds[:, c].copy()
            
            for tau in candidate_tau:
                col_pred = (y_probs[:, c] >= tau).astype(np.int32)
                
                # Fast single-class F1 check
                tp = np.sum((col_pred == 1) & (y_true[:, c] == 1))
                fp = np.sum((col_pred == 1) & (y_true[:, c] == 0))
                fn = np.sum((col_pred == 0) & (y_true[:, c] == 1))
                denom = 2 * tp + fp + fn
                f1_val = (2 * tp) / denom if denom > 0 else 0.0
                
                if f1_val > best_f1_c:
                    best_f1_c = f1_val
                    best_tau_c = tau
            
            if best_tau_c != thresholds[c]:
                thresholds[c] = best_tau_c
                current_preds[:, c] = (y_probs[:, c] >= best_tau_c).astype(np.int32)
                current_f1s[c] = best_f1_c
                improved = True
                
        new_macro_f1 = np.mean(current_f1s)
        print(f"  Iteration {iteration + 1}/{max_iters} complete: Macro-F1 = {new_macro_f1:.4f}")
        if not improved:
            break
            
    return thresholds, np.mean(current_f1s)


def main():
    np.random.seed(42)
    print("=== Phase 5: Multi-Label Coordinate Ascent Threshold Calibration ===")
    
    num_samples = 1200
    num_classes = 20  # Scaled for quick demonstration (represents the 100 classes)
    
    # Simulate power-law class distribution (legal statutes: few frequent, many rare)
    class_frequencies = np.exp(-0.25 * np.arange(num_classes))
    class_frequencies = class_frequencies / np.max(class_frequencies) * 0.45
    class_frequencies = np.clip(class_frequencies, 0.015, 0.45) # rare classes ~1.5% frequency
    
    y_true = np.zeros((num_samples, num_classes), dtype=np.int32)
    for c in range(num_classes):
        y_true[:, c] = (np.random.rand(num_samples) < class_frequencies[c]).astype(np.int32)
        
    # Simulate calibrated model probabilities with noise
    y_probs = np.zeros((num_samples, num_classes), dtype=np.float32)
    for c in range(num_classes):
        signal = y_true[:, c] * 0.40 + 0.10
        noise = np.random.normal(0, 0.15, num_samples)
        y_probs[:, c] = np.clip(signal + noise, 0.001, 0.999)
        
    print(f"Dataset: {num_samples} documents across {num_classes} legal statute classes")
    print(f"Most frequent class positive samples: {np.sum(y_true[:, 0])}")
    print(f"Rarest class positive samples:        {np.sum(y_true[:, -1])}\n")
    
    # 1. Baseline with standard threshold tau = 0.50
    preds_default = (y_probs >= 0.50).astype(np.int32)
    f1_per_class_default = compute_per_class_f1(y_true, preds_default)
    zero_pred_classes = np.sum(np.sum(preds_default, axis=0) == 0)
    macro_f1_default = np.mean(f1_per_class_default)
    
    print(f"Standard Threshold Results (tau = 0.50 for all classes):")
    print(f"  Macro-F1: {macro_f1_default:.4f}")
    print(f"  Classes with zero predicted positives: {zero_pred_classes} of {num_classes}")
    print(f"  Rare class #{num_classes-1} F1: {f1_per_class_default[-1]:.4f}\n")
    
    # 2. Run Coordinate Ascent Optimization
    optimal_thresholds, macro_f1_opt = coordinate_ascent_threshold_search(y_true, y_probs)
    
    preds_opt = (y_probs >= optimal_thresholds).astype(np.int32)
    f1_per_class_opt = compute_per_class_f1(y_true, preds_opt)
    zero_pred_opt = np.sum(np.sum(preds_opt, axis=0) == 0)
    
    print(f"\nCalibrated Threshold Results:")
    print(f"  Macro-F1 after Coordinate Ascent: {macro_f1_opt:.4f}")
    print(f"  Delta Gain: +{(macro_f1_opt - macro_f1_default)*100:.2f}% Macro-F1 points")
    print(f"  Classes with zero positive predictions: {zero_pred_opt} of {num_classes}")
    print(f"  Rare class #{num_classes-1} threshold lowered to: {optimal_thresholds[-1]:.3f}")
    print(f"  Rare class #{num_classes-1} F1 improved to: {f1_per_class_opt[-1]:.4f}")
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
