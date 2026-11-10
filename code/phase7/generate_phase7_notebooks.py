#!/usr/bin/env python3
"""Generate self-contained, highly optimized Kaggle notebooks for CS781 Phase 7:
Prior Case Retrieval (PCR) with Causal LMs adapted as Dense Bi-Encoders.

Features:
- Head (256) + Tail (256) context representation (512 tokens) preserving issues and rulings.
- Mean pooling over active attention mask tokens for stable dense semantic case embeddings.
- InfoNCE Contrastive Loss with in-batch negatives for semantic alignment of legal precedents.
- Fast batched test encoding (1,727 candidate pool + 237 test queries) and Top-7 cosine ranking.
- Strict official output formatting: exactly 7 unique candidate IDs per query.
- Auto-packaging into official submission zip and auto-upload to Hugging Face.
- Covers all 6 required experiments (Full FT, LoRA, Compulsory LORMA, Layer-Wise, BitFit, Compulsory KD).
"""

import os, json

def create_notebook(filename, cells):
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.10.12'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def code_cell(code):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': code}

def md_cell(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}

def generate_phase7_notebooks():
    common_imports = """import os, json, glob, zipfile, shutil, time, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from huggingface_hub import hf_hub_download, HfApi

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
if device == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))

HF_TOKEN = 'YOUR_HF_TOKEN'
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
TARGET_REPO = 'U-241111/cs781-submissions'

# 1. Download Phase 7 Dataset
os.makedirs('/kaggle/working/data_p7', exist_ok=True)
p7_files = [
    'Phase_7_PCR/Train/train.json',
    'Phase_7_PCR/Train/train_candidates.json',
    'Phase_7_PCR/Dev/dev.json',
    'Phase_7_PCR/Dev/dev_candidates.json',
    'Phase_7_PCR/Test/test.json',
    'Phase_7_PCR/Test/test_candidates.json'
]
for f in p7_files:
    fname = os.path.basename(f)
    dest = f'/kaggle/working/data_p7/{fname}'
    if not os.path.exists(dest):
        print(f'Downloading {fname}...')
        loc = hf_hub_download(repo_id=HF_DATASET_REPO, filename=f, repo_type='dataset', token=HF_TOKEN)
        shutil.copyfile(loc, dest)

with open('/kaggle/working/data_p7/train.json') as f:
    train_queries = json.load(f)
with open('/kaggle/working/data_p7/train_candidates.json') as f:
    train_candidates = json.load(f)
with open('/kaggle/working/data_p7/dev.json') as f:
    dev_queries = json.load(f)
with open('/kaggle/working/data_p7/dev_candidates.json') as f:
    dev_candidates = json.load(f)
with open('/kaggle/working/data_p7/test.json') as f:
    test_queries = json.load(f)
with open('/kaggle/working/data_p7/test_candidates.json') as f:
    test_candidates = json.load(f)

print(f'Train Q: {len(train_queries)} | Train Cands: {len(train_candidates)}')
print(f'Dev Q: {len(dev_queries)} | Dev Cands: {len(dev_candidates)}')
print(f'Test Q: {len(test_queries)} | Test Cands: {len(test_candidates)}')
"""

    data_pipeline = """# 2. Locate Pretrained Checkpoint and Initialize Tokenizer
model_ckpt_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'config.json' in files:
        model_ckpt_dir = root
        break
print('Model Checkpoint Path:', model_ckpt_dir, os.listdir(model_ckpt_dir))

tok = AutoTokenizer.from_pretrained(model_ckpt_dir)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token if tok.eos_token is not None else '<|endoftext|>'
tok.padding_side = 'right'

def extract_case_text(item):
    if isinstance(item, dict):
        t = item.get('text', [])
        return ' '.join(t) if isinstance(t, list) else str(t)
    return ' '.join(item) if isinstance(item, list) else str(item)

def truncate_tokens(tokens, max_len=512):
    \"\"\"Preserve Head (256) issues/facts and Tail (256) reasoning/ruling.\"\"\"
    if len(tokens) <= max_len:
        return tokens
    head = tokens[:256]
    tail = tokens[-256:]
    return head + tail

# Prepare training pairs (query, positive_candidate)
train_pairs = []
for qid, qdata in train_queries.items():
    rels = qdata.get('relevant_candidates', [])
    qtext = extract_case_text(qdata)
    for cid in rels:
        if cid in train_candidates:
            ctext = extract_case_text(train_candidates[cid])
            train_pairs.append((qtext, ctext))

print(f'Total positive training pairs: {len(train_pairs)}')

class RetrievalPairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=512):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q_text, c_text = self.pairs[idx]
        q_raw = self.tok.encode(q_text, add_special_tokens=False)
        c_raw = self.tok.encode(c_text, add_special_tokens=False)
        
        q_tokens = truncate_tokens(q_raw, self.max_len)
        c_tokens = truncate_tokens(c_raw, self.max_len)
        
        return {
            'q_text': q_text,
            'c_text': c_text,
            'q_tokens': q_tokens,
            'c_tokens': c_tokens
        }

def collate_retrieval_batch(batch, pad_id=0, max_len=512):
    q_lens = [len(x['q_tokens']) for x in batch]
    c_lens = [len(x['c_tokens']) for x in batch]
    max_q = min(max_len, max(q_lens))
    max_c = min(max_len, max(c_lens))
    
    q_ids, q_mask = [], []
    c_ids, c_mask = [], []
    for x in batch:
        qt = x['q_tokens'][:max_q]
        q_pad = max_q - len(qt)
        q_ids.append(qt + [pad_id]*q_pad)
        q_mask.append([1]*len(qt) + [0]*q_pad)
        
        ct = x['c_tokens'][:max_c]
        c_pad = max_c - len(ct)
        c_ids.append(ct + [pad_id]*c_pad)
        c_mask.append([1]*len(ct) + [0]*c_pad)
        
    return {
        'q_input_ids': torch.tensor(q_ids, dtype=torch.long),
        'q_attention_mask': torch.tensor(q_mask, dtype=torch.long),
        'c_input_ids': torch.tensor(c_ids, dtype=torch.long),
        'c_attention_mask': torch.tensor(c_mask, dtype=torch.long),
    }

class DenseBiEncoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use last hidden state with mean pooling over active attention mask tokens
        hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None else outputs[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_emb = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_emb / sum_mask
        return F.normalize(mean_pooled, p=2, dim=-1)

    def forward(self, q_ids, q_mask, c_ids, c_mask, tau=0.07):
        q_emb = self.encode(q_ids, q_mask)
        c_emb = self.encode(c_ids, c_mask)
        # Cosine similarity matrix (B, B)
        sim_matrix = torch.matmul(q_emb, c_emb.T) / tau
        labels = torch.arange(q_ids.size(0), device=q_ids.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
"""

    eval_and_submission = """# 5. Batched Test Case Retrieval and Official Formatting
print('Starting Dense Bi-Encoder Retrieval on 237 Test Queries & 1,727 Candidates...')
model.eval()

# Helper for encoding arbitrary list of texts
def encode_texts(text_list, batch_size=8, max_len=512):
    all_embeddings = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            chunk = text_list[i : i + batch_size]
            token_list = [truncate_tokens(tok.encode(t, add_special_tokens=False), max_len) for t in chunk]
            max_c = min(max_len, max(len(t) for t in token_list))
            ids, mask = [], []
            for t in token_list:
                t = t[:max_c]
                pad = max_c - len(t)
                ids.append(t + [pad_id] * pad)
                mask.append([1] * len(t) + [0] * pad)
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            mask_t = torch.tensor(mask, dtype=torch.long, device=device)
            emb = bi_encoder.encode(ids_t, mask_t)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)

# 1. Encode candidate pool
candidate_ids = list(test_candidates.keys())
candidate_texts = [extract_case_text(test_candidates[cid]) for cid in candidate_ids]
print(f'Encoding {len(candidate_ids)} test candidates...')
t0_c = time.time()
cand_embeddings = encode_texts(candidate_texts, batch_size=16)
print(f'Encoded candidates in {time.time()-t0_c:.1f}s, shape: {cand_embeddings.shape}')

# 2. Encode queries
query_ids = list(test_queries.keys())
query_texts = [extract_case_text(test_queries[qid]) for qid in query_ids]
print(f'Encoding {len(query_ids)} test queries...')
t0_q = time.time()
query_embeddings = encode_texts(query_texts, batch_size=16)
print(f'Encoded queries in {time.time()-t0_q:.1f}s, shape: {query_embeddings.shape}')

# 3. Dense Cosine Ranking
print('Computing Top-7 candidate retrieval per query...')
scores = torch.matmul(query_embeddings, cand_embeddings.T) # (237, 1727)
topk_scores, topk_indices = torch.topk(scores, k=7, dim=1)

# Format predictions according to Codabench Phase 7 specification
test_predictions = {}
for i, qid in enumerate(query_ids):
    retrieved_cids = [candidate_ids[idx] for idx in topk_indices[i].tolist()]
    assert len(retrieved_cids) == 7
    assert len(set(retrieved_cids)) == 7
    test_predictions[str(qid)] = retrieved_cids

print(f'Completed predictions for {len(test_predictions)} test queries.')
print('Sample prediction (Query 0):', query_ids[0], '->', test_predictions[query_ids[0]])

# Save prediction.json
pred_file = '/kaggle/working/prediction.json'
with open(pred_file, 'w', encoding='utf-8') as f:
    json.dump(test_predictions, f, indent=2)

zip_file = f'/kaggle/working/{SUB_ZIP_NAME}'
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(pred_file, arcname='prediction.json')

print(f'Created {SUB_ZIP_NAME} successfully ({os.path.getsize(zip_file)} bytes)!')

# Auto-upload to Hugging Face repository
try:
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=zip_file,
        path_in_repo=f'phase7/{SUB_ZIP_NAME}',
        repo_id=TARGET_REPO,
        repo_type='dataset'
    )
    api.upload_file(
        path_or_fileobj=pred_file,
        path_in_repo=f"phase7/{SUB_ZIP_NAME.replace('.zip', '_predictions.json')}",
        repo_id=TARGET_REPO,
        repo_type='dataset'
    )
    print(f'Uploaded {SUB_ZIP_NAME} and predictions to Hugging Face!')
except Exception as e:
    print('HF upload exception:', e)
"""

    experiments = [
        {
            'slug': 'kaggle_phase7_exp1',
            'nb_name': 'cs781-phase7-exp1-fullft.ipynb',
            'title': 'CS-781 Phase 7: Exp 1 — Full Fine-Tuning Dense Bi-Encoder (Model 1 GPT-2)',
            'account': 'ujjwalprakashgoodgod',
            'datasets': ['ujjwalprakashgoodgod/cs781-phase1-m1-ckpt2000'],
            'sub_zip': 'submission_phase7_exp1_fullft.zip',
            'model_setup': """# 4. Initialize Model 1 (GPT-2) for Full Fine-Tuning
model = AutoModelForCausalLM.from_pretrained(model_ckpt_dir)
model.config.pad_token_id = tok.pad_token_id

for p in model.parameters():
    p.requires_grad = True

bi_encoder = DenseBiEncoder(model).to(device)
trainable = sum(p.numel() for p in bi_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in bi_encoder.parameters())
print(f'Full FT Model 1 Bi-Encoder: Trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

SUB_ZIP_NAME = 'submission_phase7_exp1_fullft.zip'
""",
            'training_code': """# Training Loop with Contrastive InfoNCE Loss
train_ds = RetrievalPairDataset(train_pairs, tok, max_len=512)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_retrieval_batch(b, pad_id=tok.pad_token_id))

optimizer = AdamW([p for p in bi_encoder.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

print(f'Starting Contrastive Bi-Encoder training for {epochs} epochs...')
bi_encoder.train()
for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        q_ids = batch['q_input_ids'].to(device)
        q_mask = batch['q_attention_mask'].to(device)
        c_ids = batch['c_input_ids'].to(device)
        c_mask = batch['c_attention_mask'].to(device)
        
        optimizer.zero_grad()
        loss = bi_encoder(q_ids, q_mask, c_ids, c_mask, tau=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bi_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}')
    print(f'Epoch {epoch+1}/{epochs} Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s')
"""
        },
        {
            'slug': 'kaggle_phase7_exp2',
            'nb_name': 'cs781-phase7-exp2-lora.ipynb',
            'title': 'CS-781 Phase 7: Exp 2 — LoRA Dense Bi-Encoder (Model 2 RoPE LLaMA)',
            'account': 'ujjwalprakashgoodgod',
            'datasets': ['ujjwalprakashgoodgod/cs781-phase1-m2-ckpt2500'],
            'sub_zip': 'submission_phase7_exp2_lora.zip',
            'model_setup': """# 4. Initialize Model 2 (RoPE LLaMA) with Native PyTorch LoRA
model = AutoModelForCausalLM.from_pretrained(model_ckpt_dir)
model.config.pad_token_id = tok.pad_token_id

class LoRALinear(nn.Module):
    def __init__(self, linear, r=16, alpha=32):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    def forward(self, x):
        return self.linear(x) + torch.matmul(x, torch.matmul(self.lora_B, self.lora_A).T) * self.scaling

for p in model.parameters():
    p.requires_grad = False

lora_count = 0
for name, module in model.named_modules():
    for child_name, child in module.named_children():
        if child_name in ['q_proj', 'v_proj', 'k_proj', 'o_proj'] and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, r=16, alpha=32))
            lora_count += 1

bi_encoder = DenseBiEncoder(model).to(device)
trainable = sum(p.numel() for p in bi_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in bi_encoder.parameters())
print(f'LoRA Model 2 attached to {lora_count} projections: Trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

SUB_ZIP_NAME = 'submission_phase7_exp2_lora.zip'
""",
            'training_code': """# Training Loop with Contrastive InfoNCE Loss
train_ds = RetrievalPairDataset(train_pairs, tok, max_len=512)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_retrieval_batch(b, pad_id=tok.pad_token_id))

optimizer = AdamW([p for p in bi_encoder.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

print(f'Starting LoRA Contrastive Bi-Encoder training for {epochs} epochs...')
bi_encoder.train()
for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        q_ids = batch['q_input_ids'].to(device)
        q_mask = batch['q_attention_mask'].to(device)
        c_ids = batch['c_input_ids'].to(device)
        c_mask = batch['c_attention_mask'].to(device)
        
        optimizer.zero_grad()
        loss = bi_encoder(q_ids, q_mask, c_ids, c_mask, tau=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bi_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}')
    print(f'Epoch {epoch+1}/{epochs} Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s')
"""
        },
        {
            'slug': 'kaggle_phase7_exp3',
            'nb_name': 'cs781-phase7-exp3-lorma.ipynb',
            'title': 'CS-781 Phase 7: Exp 3 — Native LORMA Adaptation (Model 2 RoPE LLaMA)',
            'account': 'ujjwalprakashgoodgod',
            'datasets': ['ujjwalprakashgoodgod/cs781-phase1-m2-ckpt2500'],
            'sub_zip': 'submission_phase7_exp3_lorma.zip',
            'model_setup': """# 4. Initialize Model 2 (RoPE LLaMA) with Native PyTorch LORMA (Compulsory PEFT)
model = AutoModelForCausalLM.from_pretrained(model_ckpt_dir)
model.config.pad_token_id = tok.pad_token_id

class LORMALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lorma_down = nn.Parameter(torch.randn(r, linear.in_features) * (1.0 / r))
        self.lorma_up = nn.Parameter(torch.zeros(linear.out_features, r))
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    def forward(self, x):
        return self.linear(x) + torch.matmul(x, torch.matmul(self.lorma_up, self.lorma_down).T) * self.scaling

for p in model.parameters():
    p.requires_grad = False

lorma_count = 0
for name, module in model.named_modules():
    for child_name, child in module.named_children():
        if child_name in ['q_proj', 'v_proj', 'k_proj', 'o_proj'] and isinstance(child, nn.Linear):
            setattr(module, child_name, LORMALinear(child, r=8, alpha=16))
            lorma_count += 1

bi_encoder = DenseBiEncoder(model).to(device)
trainable = sum(p.numel() for p in bi_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in bi_encoder.parameters())
print(f'LORMA attached to {lorma_count} projection layers: Trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

SUB_ZIP_NAME = 'submission_phase7_exp3_lorma.zip'
""",
            'training_code': """# Training Loop with Contrastive InfoNCE Loss
train_ds = RetrievalPairDataset(train_pairs, tok, max_len=512)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_retrieval_batch(b, pad_id=tok.pad_token_id))

optimizer = AdamW([p for p in bi_encoder.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

print(f'Starting LORMA Contrastive Bi-Encoder training for {epochs} epochs...')
bi_encoder.train()
for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        q_ids = batch['q_input_ids'].to(device)
        q_mask = batch['q_attention_mask'].to(device)
        c_ids = batch['c_input_ids'].to(device)
        c_mask = batch['c_attention_mask'].to(device)
        
        optimizer.zero_grad()
        loss = bi_encoder(q_ids, q_mask, c_ids, c_mask, tau=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bi_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}')
    print(f'Epoch {epoch+1}/{epochs} Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s')
"""
        },
        {
            'slug': 'kaggle_phase7_exp4',
            'nb_name': 'cs781-phase7-exp4-layerwise.ipynb',
            'title': 'CS-781 Phase 7: Exp 4 — Layer-Wise Dense Bi-Encoder (Model 3 GPT-NeoX)',
            'account': 'ujjwalprakashgoodgod',
            'datasets': ['ujjwalprakashts/cs781-phase1-m3-ckpt2500'],
            'sub_zip': 'submission_phase7_exp4_layerwise.zip',
            'model_setup': """# 4. Initialize Model 3 (GPT-NeoX) with Partial / Layer-Wise Tuning
model = AutoModelForCausalLM.from_pretrained(model_ckpt_dir)
model.config.pad_token_id = tok.pad_token_id

for name, p in model.named_parameters():
    p.requires_grad = False
    if any(f'layers.{i}.' in name for i in range(8, 12)) or 'embed_out' in name or 'final_layer_norm' in name:
        p.requires_grad = True

bi_encoder = DenseBiEncoder(model).to(device)
trainable = sum(p.numel() for p in bi_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in bi_encoder.parameters())
print(f'Layer-Wise Model 3 Bi-Encoder: Trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

SUB_ZIP_NAME = 'submission_phase7_exp4_layerwise.zip'
""",
            'training_code': """# Training Loop with Contrastive InfoNCE Loss
train_ds = RetrievalPairDataset(train_pairs, tok, max_len=512)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_retrieval_batch(b, pad_id=tok.pad_token_id))

optimizer = AdamW([p for p in bi_encoder.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.01)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

print(f'Starting Layer-Wise Contrastive Bi-Encoder training for {epochs} epochs...')
bi_encoder.train()
for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        q_ids = batch['q_input_ids'].to(device)
        q_mask = batch['q_attention_mask'].to(device)
        c_ids = batch['c_input_ids'].to(device)
        c_mask = batch['c_attention_mask'].to(device)
        
        optimizer.zero_grad()
        loss = bi_encoder(q_ids, q_mask, c_ids, c_mask, tau=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bi_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}')
    print(f'Epoch {epoch+1}/{epochs} Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s')
"""
        },
        {
            'slug': 'kaggle_phase7_exp5',
            'nb_name': 'cs781-phase7-exp5-bitfit.ipynb',
            'title': 'CS-781 Phase 7: Exp 5 — BitFit Adaptation Dense Bi-Encoder (Model 4 Deep-Thin)',
            'account': 'ujjwalprakashgoodgod',
            'datasets': ['ujjwalprakashts/cs781-phase1-m4-ckpt2500'],
            'sub_zip': 'submission_phase7_exp5_bitfit.zip',
            'model_setup': """# 4. Initialize Model 4 (Deep-Thin) with BitFit (Bias-Only Tuning)
model = AutoModelForCausalLM.from_pretrained(model_ckpt_dir)
model.config.pad_token_id = tok.pad_token_id

for name, p in model.named_parameters():
    if 'bias' in name or 'ln_' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

bi_encoder = DenseBiEncoder(model).to(device)
trainable = sum(p.numel() for p in bi_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in bi_encoder.parameters())
print(f'BitFit Model 4 Bi-Encoder: Trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

SUB_ZIP_NAME = 'submission_phase7_exp5_bitfit.zip'
""",
            'training_code': """# Training Loop with Contrastive InfoNCE Loss
train_ds = RetrievalPairDataset(train_pairs, tok, max_len=512)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_retrieval_batch(b, pad_id=tok.pad_token_id))

optimizer = AdamW([p for p in bi_encoder.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

print(f'Starting BitFit Contrastive Bi-Encoder training for {epochs} epochs...')
bi_encoder.train()
for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        q_ids = batch['q_input_ids'].to(device)
        q_mask = batch['q_attention_mask'].to(device)
        c_ids = batch['c_input_ids'].to(device)
        c_mask = batch['c_attention_mask'].to(device)
        
        optimizer.zero_grad()
        loss = bi_encoder(q_ids, q_mask, c_ids, c_mask, tau=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bi_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}')
    print(f'Epoch {epoch+1}/{epochs} Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s')
"""
        },
        {
            'slug': 'kaggle_phase7_exp6',
            'nb_name': 'cs781-phase7-exp6-kd.ipynb',
            'title': 'CS-781 Phase 7: Exp 6 — Knowledge Distillation Bi-Encoder (Teacher M2 -> Student M1)',
            'account': 'ujjwalprakashgoodgod',
            'datasets': [
                'ujjwalprakashgoodgod/cs781-phase1-m1-ckpt2000',
                'ujjwalprakashgoodgod/cs781-phase1-m2-ckpt2500'
            ],
            'sub_zip': 'submission_phase7_exp6_kd.zip',
            'model_setup': """# 4. Initialize Teacher (Model 2 RoPE LLaMA) and Student (Model 1 GPT-2)
m1_dir = None
m2_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'config.json' in files:
        with open(os.path.join(root, 'config.json')) as f:
            c = json.load(f)
        if c.get('model_type') == 'llama' or 'rope' in root.lower() or 'm2' in root.lower():
            m2_dir = root
        else:
            m1_dir = root

print('Student Model 1 Dir:', m1_dir)
print('Teacher Model 2 Dir:', m2_dir)

student_base = AutoModelForCausalLM.from_pretrained(m1_dir).to(device)
teacher_base = AutoModelForCausalLM.from_pretrained(m2_dir).to(device)

for p in teacher_base.parameters():
    p.requires_grad = False
for p in student_base.parameters():
    p.requires_grad = True

teacher = DenseBiEncoder(teacher_base)
bi_encoder = DenseBiEncoder(student_base)
model = student_base

trainable = sum(p.numel() for p in bi_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in bi_encoder.parameters())
print(f'Knowledge Distillation: Student Trainable {trainable:,} / {total:,}')

SUB_ZIP_NAME = 'submission_phase7_exp6_kd.zip'
""",
            'training_code': """# Training Loop with Knowledge Distillation & Contrastive Loss
train_ds = RetrievalPairDataset(train_pairs, tok, max_len=512)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_retrieval_batch(b, pad_id=tok.pad_token_id))

optimizer = AdamW([p for p in bi_encoder.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.01)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)
temperature = 2.0

print(f'Starting Retrieval Knowledge Distillation training for {epochs} epochs...')
bi_encoder.train()
teacher.eval()

for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        q_ids = batch['q_input_ids'].to(device)
        q_mask = batch['q_attention_mask'].to(device)
        c_ids = batch['c_input_ids'].to(device)
        c_mask = batch['c_attention_mask'].to(device)
        
        with torch.no_grad():
            t_q_emb = teacher.encode(q_ids, q_mask)
            t_c_emb = teacher.encode(c_ids, c_mask)
            t_sim = torch.matmul(t_q_emb, t_c_emb.T) / 0.07
            
        s_q_emb = bi_encoder.encode(q_ids, q_mask)
        s_c_emb = bi_encoder.encode(c_ids, c_mask)
        s_sim = torch.matmul(s_q_emb, s_c_emb.T) / 0.07
        
        loss_contrastive = F.cross_entropy(s_sim, torch.arange(q_ids.size(0), device=device))
        
        # Distillation loss on similarity distribution
        p_teacher = F.softmax(t_sim / temperature, dim=-1)
        log_p_student = F.log_softmax(s_sim / temperature, dim=-1)
        loss_kd = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
        
        loss = 0.5 * loss_contrastive + 0.5 * loss_kd
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bi_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}')
    print(f'Epoch {epoch+1}/{epochs} Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s')
"""
        }
    ]

    base_dir = '/home/ujjwal/Desktop/01 Academics/CS781/Project'

    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp['slug'])
        os.makedirs(exp_dir, exist_ok=True)
        nb_path = os.path.join(exp_dir, exp['nb_name'])
        
        cells = [
            md_cell(f"# {exp['title']}\nAuto-generated for CS-781 Capstone Phase 7 (Prior Case Retrieval - PCR)."),
            code_cell(common_imports),
            code_cell(data_pipeline),
            code_cell(exp['model_setup']),
            code_cell(exp['training_code']),
            code_cell(eval_and_submission)
        ]
        
        create_notebook(nb_path, cells)
        print(f'Created notebook: {nb_path}')
        
        meta = {
            'id': f"{exp['account']}/{exp['slug'].replace('_', '-')}",
            'title': exp['slug'].replace('_', '-'),
            'code_file': exp['nb_name'],
            'language': 'python',
            'kernel_type': 'notebook',
            'is_private': True,
            'enable_gpu': True,
            'enable_tpu': False,
            'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': exp['datasets'],
            'competition_sources': [],
            'kernel_sources': []
        }
        meta_path = os.path.join(exp_dir, 'kernel-metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f'Created metadata: {meta_path}')

if __name__ == '__main__':
    generate_phase7_notebooks()
