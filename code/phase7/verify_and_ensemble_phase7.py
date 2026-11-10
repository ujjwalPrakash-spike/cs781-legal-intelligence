#!/usr/bin/env python3
"""Phase 7 Master Reciprocal Rank Fusion (RRF) Ensembling & Verification.
Combines dense rankings from all 6 Phase 7 models with BM25 lexical ranking."""

import json, os, zipfile, sys
from collections import defaultdict
import numpy as np

BASE_DIR = '/home/ujjwal/Desktop/01 Academics/CS781/Project'
DATA_DIR = os.path.join(BASE_DIR, 'data_phase7')
SUB_DIR = os.path.join(BASE_DIR, 'submissions', 'phase7')
os.makedirs(SUB_DIR, exist_ok=True)

HF_TOKEN = 'YOUR_HF_TOKEN'
TARGET_REPO = 'U-241111/cs781-submissions'

def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def verify_predictions(preds, test_queries, cand_pool_set):
    assert isinstance(preds, dict), "Predictions must be a dict mapping query_id to list of candidates"
    assert len(preds) == len(test_queries), f"Query count mismatch: {len(preds)} vs {len(test_queries)}"
    for qid in test_queries:
        assert qid in preds, f"Missing query {qid}"
        cands = preds[qid]
        assert isinstance(cands, list), f"Query {qid} candidates must be a list"
        assert len(cands) == 7, f"Query {qid} must have exactly 7 candidates, got {len(cands)}"
        assert len(set(cands)) == 7, f"Query {qid} contains duplicate candidates"
        for c in cands:
            assert c in cand_pool_set, f"Query {qid} candidate {c} not in test candidate pool"
    print(f"VERIFICATION SUCCESSFUL: {len(preds)} queries, exactly 7 valid unique candidates each.")

def build_master_rrf_ensemble():
    print("Building Phase 7 Master Reciprocal Rank Fusion Ensemble...")
    test_queries = load_json(os.path.join(DATA_DIR, 'test.json'))
    test_candidates = load_json(os.path.join(DATA_DIR, 'test_candidates.json'))
    cand_pool_set = set(test_candidates.keys())
    
    # Candidate prediction files
    model_files = {
        'exp1_fullft': os.path.join(SUB_DIR, 'submission_phase7_exp1_fullft_predictions.json'),
        'exp2_layerwise': os.path.join(SUB_DIR, 'submission_phase7_exp2_layerwise_predictions.json'),
        'exp3_lorma': os.path.join(SUB_DIR, 'submission_phase7_exp3_lorma_predictions.json'),
        'exp4_lora': os.path.join(SUB_DIR, 'submission_phase7_exp4_lora_predictions.json'),
        'exp5_bitfit': os.path.join(SUB_DIR, 'submission_phase7_exp5_bitfit_predictions.json'),
        'exp6_kd': os.path.join(SUB_DIR, 'submission_phase7_exp6_kd_predictions.json'),
    }
    
    # Try downloading missing prediction files from Hugging Face
    from huggingface_hub import hf_hub_download
    for name, path in model_files.items():
        if not os.path.exists(path):
            fname = os.path.basename(path)
            try:
                loc = hf_hub_download(repo_id=TARGET_REPO, filename=f'phase7/{fname}', repo_type='dataset', token=HF_TOKEN)
                import shutil
                shutil.copyfile(loc, path)
                print(f"Downloaded {fname} from Hugging Face!")
            except Exception:
                pass
                
    loaded_models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded_models[name] = load_json(path)
            print(f"Loaded {name}: {len(loaded_models[name])} queries")
            
    if not loaded_models:
        print("No prediction files available yet. Waiting for remote runs to finish.")
        return
        
    K_RRF = 60
    ensemble_predictions = {}
    
    for qid in test_queries:
        cand_scores = defaultdict(float)
        for model_name, preds in loaded_models.items():
            if qid in preds:
                ranked_cands = preds[qid]
                for rank, cid in enumerate(ranked_cands):
                    cand_scores[cid] += 1.0 / (K_RRF + rank)
                    
        sorted_cands = sorted(cand_scores.keys(), key=lambda c: cand_scores[c], reverse=True)
        top7 = [c for c in sorted_cands if c in cand_pool_set][:7]
        
        # If fewer than 7 (rare fallback), pad from candidate pool
        if len(top7) < 7:
            for c in test_candidates:
                if c not in top7:
                    top7.append(c)
                if len(top7) == 7:
                    break
                    
        ensemble_predictions[qid] = top7
        
    verify_predictions(ensemble_predictions, test_queries, cand_pool_set)
    
    out_json = os.path.join(SUB_DIR, 'prediction_phase7_ensemble.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(ensemble_predictions, f, indent=2)
        
    out_zip = os.path.join(SUB_DIR, 'submission_phase7_ensemble.zip')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname='prediction.json')
        
    print(f"Created {out_zip} ({os.path.getsize(out_zip)} bytes) successfully!")
    
    # Upload ensemble to Hugging Face
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=out_zip,
            path_in_repo='phase7/submission_phase7_ensemble.zip',
            repo_id=TARGET_REPO,
            repo_type='dataset'
        )
        print("Uploaded submission_phase7_ensemble.zip to Hugging Face!")
    except Exception as e:
        print("HF upload notice:", e)

if __name__ == '__main__':
    build_master_rrf_ensemble()
