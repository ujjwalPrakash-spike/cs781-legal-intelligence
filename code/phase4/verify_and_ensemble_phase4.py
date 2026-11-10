#!/usr/bin/env python3
"""Local verification and ensemble builder for Phase 4 (CJPE) - Pure Python (zero dependencies)."""
import os, json, zipfile, glob

BASE_DIR = '/home/ujjwal/Desktop/01 Academics/CS781/Project'
TEST_PATH = os.path.join(BASE_DIR, 'data_phase4/test.json')
SUB_DIR = os.path.join(BASE_DIR, 'submissions/phase4')
os.makedirs(SUB_DIR, exist_ok=True)

def verify_submission(zip_path, expected_ids):
    name = os.path.basename(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = z.namelist()
        if 'prediction.json' not in files:
            raise ValueError(f'{name}: Missing prediction.json at root of ZIP!')
        data = json.loads(z.read('prediction.json'))

    if not isinstance(data, list) or not data:
        raise ValueError(f'{name}: prediction.json must be a non-empty list of dicts!')

    preds_map = {}
    for item in data:
        if 'id' not in item or 'pred' not in item:
            raise ValueError(f'{name}: Item missing id or pred: {item}')
        if item['pred'] not in (0, 1):
            raise ValueError(f'{name}: Invalid pred value: {item["pred"]}')
        preds_map[str(item['id'])] = item['pred']

    if set(preds_map.keys()) != set(expected_ids):
        diff = set(expected_ids) - set(preds_map.keys())
        extra = set(preds_map.keys()) - set(expected_ids)
        raise ValueError(f'{name}: ID mismatch! Missing: {len(diff)}, Extra: {len(extra)}')

    ones = sum(preds_map.values())
    zeros = len(preds_map) - ones
    print(f'✅ {name}: VERIFIED! (1517 test cases, Class 0: {zeros} ({100*zeros/len(preds_map):.1f}%), Class 1: {ones} ({100*ones/len(preds_map):.1f}%))')
    return preds_map

def main():
    with open(TEST_PATH) as f:
        test_data = json.load(f)
    expected_ids = [str(d['id']) for d in test_data]
    print(f'Total expected test cases: {len(expected_ids)}')

    zips = sorted(glob.glob(os.path.join(SUB_DIR, '*.zip')))
    print(f'Found {len(zips)} submissions in {SUB_DIR}')
    for zp in zips:
        try:
            verify_submission(zp, expected_ids)
        except Exception as e:
            print(f'❌ {os.path.basename(zp)}: {e}')

if __name__ == '__main__':
    main()
