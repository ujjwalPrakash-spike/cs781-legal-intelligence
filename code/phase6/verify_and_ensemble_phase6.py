#!/usr/bin/env python3
"""Phase 6 Verification, Evaluation, and Minimum Bayes Risk (MBR) Consensus Ensembling.
Strictly ensures 100 test document IDs are present and computes cross-model medoids.
"""

import os, json, glob, zipfile
from collections import Counter

TEST_PATH = '/home/ujjwal/Desktop/01 Academics/CS781/Project/data_phase6/test.json'
SUBMISSIONS_DIR = '/home/ujjwal/Desktop/01 Academics/CS781/Project/submissions/phase6'

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

def verify_submission_zip(zip_path):
    """Verify that zip file contains valid prediction.json matching 100 test cases."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"File not found: {zip_path}")
    
    with open(TEST_PATH) as f:
        test_data = json.load(f)
    expected_ids = set(str(d['id']) for d in test_data)
    
    with zipfile.ZipFile(zip_path) as z:
        if 'prediction.json' not in z.namelist():
            raise ValueError(f"{zip_path} missing prediction.json!")
        data = json.loads(z.read('prediction.json').decode('utf-8'))
        
    if isinstance(data, list):
        pred_dict = {str(r['id']): str(r['summary']).strip() for r in data if 'id' in r and 'summary' in r}
    elif isinstance(data, dict):
        pred_dict = {str(k): str(v).strip() for k, v in data.items()}
    else:
        raise ValueError("Invalid format in prediction.json")
        
    found_ids = set(pred_dict.keys())
    missing = expected_ids - found_ids
    extra = found_ids - expected_ids
    
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={len(missing)}, extra={len(extra)}")
        
    empty_summaries = [k for k, v in pred_dict.items() if not v or len(v) < 15]
    if empty_summaries:
        raise ValueError(f"Found empty or degenerate summaries for IDs: {empty_summaries}")
        
    lengths = [len(v.split()) for v in pred_dict.values()]
    avg_len = sum(lengths) / len(lengths)
    print(f"Verified {os.path.basename(zip_path)}: Exactly 100 valid summaries! Avg word count: {avg_len:.1f} (min: {min(lengths)}, max: {max(lengths)})")
    return pred_dict

def token_f1_similarity(s1, s2):
    """Fast, robust token-level overlap F1 score for MBR medoid selection."""
    t1 = s1.lower().split()
    t2 = s2.lower().split()
    if not t1 or not t2:
        return 0.0
    c1 = Counter(t1)
    c2 = Counter(t2)
    common = sum((c1 & c2).values())
    if common == 0:
        return 0.0
    p = common / len(t1)
    r = common / len(t2)
    return 2.0 * p * r / (p + r)

def build_mbr_ensemble(candidate_json_files, output_zip_name='submission_phase6_ensemble.zip'):
    """Minimum Bayes Risk (MBR) Consensus Medoid Selection:
    For each test document, pick the candidate summary that maximizes mutual agreement
    across all fine-tuned architectures, eliminating idiosyncratic hallucinations.
    """
    with open(TEST_PATH) as f:
        test_data = json.load(f)
    test_ids = [str(d['id']) for d in test_data]
    
    all_models_preds = []
    for fpath in candidate_json_files:
        with open(fpath) as f:
            d = json.load(f)
        if isinstance(d, list):
            pdict = {str(r['id']): str(r['summary']).strip() for r in d}
        else:
            pdict = {str(k): str(v).strip() for k, v in d.items()}
        all_models_preds.append(pdict)
        
    print(f"Loaded candidate predictions from {len(candidate_json_files)} models.")
    
    ensemble_predictions = []
    for doc_id in test_ids:
        candidates = [m[doc_id] for m in all_models_preds if doc_id in m and m[doc_id]]
        if not candidates:
            raise ValueError(f"No candidates for doc_id {doc_id}")
            
        if len(candidates) == 1:
            best_summary = candidates[0]
        else:
            # Score each candidate by its average similarity to all other candidates
            best_score = -1.0
            best_summary = candidates[0]
            for i, cand in enumerate(candidates):
                other_scores = [token_f1_similarity(cand, other) for j, other in enumerate(candidates) if i != j]
                avg_sim = sum(other_scores) / len(other_scores)
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_summary = cand
                    
        ensemble_predictions.append({
            'id': doc_id,
            'summary': best_summary
        })
        
    out_json = os.path.join(SUBMISSIONS_DIR, 'prediction_ensemble.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(ensemble_predictions, f, indent=2)
        
    out_zip = os.path.join(SUBMISSIONS_DIR, output_zip_name)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname='prediction.json')
        
    print(f"Created MBR Flagship Ensemble: {out_zip} with {len(ensemble_predictions)} cases!")
    verify_submission_zip(out_zip)
    return out_zip

if __name__ == '__main__':
    print("Phase 6 Verification & MBR Ensembling module loaded successfully.")
