#!/usr/bin/env python3
"""Phase 5 Local Evaluator, Threshold Optimizer, and Multi-Model Ensemble Blender.
Strictly verifies formatting, IDs, and maximizes Macro-F1 across 100 legal statutes.
"""

import os, json, glob, zipfile
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

DEV_PATH = '/home/ujjwal/Desktop/01 Academics/CS781/Project/data_phase5/dev.json'
TEST_PATH = '/home/ujjwal/Desktop/01 Academics/CS781/Project/data_phase5/test.json'
SUBMISSIONS_DIR = '/home/ujjwal/Desktop/01 Academics/CS781/Project/submissions/phase5'

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

def load_dev_data():
    with open(DEV_PATH) as f:
        dev_data = json.load(f)
    dev_targets = np.zeros((len(dev_data), 100), dtype=np.uint8)
    for i, d in enumerate(dev_data):
        for l in d['labels']:
            if 0 <= l < 100:
                dev_targets[i, l] = 1
    return dev_data, dev_targets

def evaluate_predictions(truth_targets, pred_targets):
    macro_f1 = f1_score(truth_targets, pred_targets, average='macro', zero_division=0)
    micro_f1 = f1_score(truth_targets, pred_targets, average='micro', zero_division=0)
    subset_acc = accuracy_score(truth_targets, pred_targets)
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'subset_acc': subset_acc
    }

def optimize_per_class_thresholds(val_probs, val_targets):
    optimal_thresholds = np.zeros(100)
    for c in range(100):
        best_f1 = 0.0
        best_th = 0.30
        for th in np.linspace(0.05, 0.60, 56):
            pred_c = (val_probs[:, c] >= th).astype(int)
            f1_c = f1_score(val_targets[:, c], pred_c, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_th = th
        optimal_thresholds[c] = best_th
    return optimal_thresholds

def build_ensemble(model_prob_files, weights=None):
    with open(TEST_PATH) as f:
        test_data = json.load(f)
    test_ids = [d['id'] for d in test_data]
    
    dev_data, dev_targets = load_dev_data()
    
    print(f'Loading {len(model_prob_files)} models for ensembling...')
    test_probs_list = []
    val_probs_list = []
    
    for val_f, test_f in model_prob_files:
        val_probs_list.append(np.load(val_f))
        test_probs_list.append(np.load(test_f))
        
    if weights is None:
        weights = [1.0 / len(test_probs_list)] * len(test_probs_list)
    weights = np.array(weights) / np.sum(weights)
    
    blended_val_probs = sum(w * p for w, p in zip(weights, val_probs_list))
    blended_test_probs = sum(w * p for w, p in zip(weights, test_probs_list))
    
    print('Optimizing per-class thresholds on blended probabilities...')
    optimal_th = optimize_per_class_thresholds(blended_val_probs, dev_targets)
    
    blended_val_preds = (blended_val_probs >= optimal_th).astype(np.uint8)
    metrics = evaluate_predictions(dev_targets, blended_val_preds)
    mf1 = metrics['macro_f1']
    micro = metrics['micro_f1']
    acc = metrics['subset_acc']
    print(f'>>> Blended Ensemble Dev Macro-F1: {mf1:.4f} | Micro-F1: {micro:.4f} | Subset Acc: {acc:.4f} <<<')
    
    final_predictions = []
    for i, did in enumerate(test_ids):
        selected = [int(c) for c in range(100) if blended_test_probs[i, c] >= optimal_th[c]]
        if not selected:
            selected = [int(np.argmax(blended_test_probs[i]))]
        final_predictions.append({
            'id': str(did),
            'pred': selected
        })
        
    out_json = os.path.join(SUBMISSIONS_DIR, 'prediction_ensemble.json')
    with open(out_json, 'w') as f:
        json.dump(final_predictions, f, indent=2)
        
    out_zip = os.path.join(SUBMISSIONS_DIR, 'submission_phase5_ensemble.zip')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname='prediction.json')
        
    print(f'Saved ensemble submission to {out_zip}!')
    return metrics

if __name__ == '__main__':
    print('Phase 5 Verification & Ensembling module ready.')
