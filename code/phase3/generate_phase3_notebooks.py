import os, json

base_dir = '/home/ujjwal/Desktop/01 Academics/CS781/Project'
os.makedirs(f'{base_dir}/submissions/phase3', exist_ok=True)

CLASS_WEIGHTS_CODE = "class_weights = torch.tensor([1.0116, 3.4927, 1.3436, 1.4491, 1.5996, 2.6390, 1.3029, 2.0223, 7.8611, 1.7279, 1.0000, 2.4442, 2.5081], device=device)"

def get_common_cell2():
    return """import os, json, glob, zipfile, shutil, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from huggingface_hub import hf_hub_download, HfApi
from sklearn.metrics import f1_score, precision_score, recall_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
if device == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))

HF_TOKEN = 'YOUR_HF_TOKEN'
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
TARGET_REPO = 'U-241111/cs781-submissions'

# 1. Download Phase 3 Dataset
os.makedirs('/kaggle/working/data_p3', exist_ok=True)
p3_files = [
    'Phase_3_RR/Train/CL_train.json',
    'Phase_3_RR/Train/CL_dev.json',
    'Phase_3_RR/Train/IT_train.json',
    'Phase_3_RR/Train/IT_dev.json',
    'Phase_3_RR/Train/label_vocab.json',
    'Phase_3_RR/Test/CL_test.json',
    'Phase_3_RR/Test/IT_test.json',
]
for f in p3_files:
    fname = os.path.basename(f)
    loc = hf_hub_download(repo_id=HF_DATASET_REPO, filename=f, repo_type='dataset', token=HF_TOKEN)
    shutil.copyfile(loc, f'/kaggle/working/data_p3/{fname}')
print('Downloaded all Phase 3 files successfully!')
"""

def get_common_cell3(exp_id):
    if exp_id == 'exp6_kd':
        return """# 2. Locate Teacher (Model 2 RoPE LLaMA) and Student (Model 1 GPT-2) Checkpoints
teacher_ckpt = None
student_ckpt = None

for root, dirs, files in os.walk('/kaggle/input'):
    if 'config.json' in files:
        if 'm2' in root.lower() or 'llama' in root.lower():
            teacher_ckpt = root
        elif 'm1' in root.lower() or 'gpt' in root.lower():
            student_ckpt = root

print('Teacher Checkpoint Path:', teacher_ckpt)
print('Student Checkpoint Path:', student_ckpt)

tok_t = AutoTokenizer.from_pretrained(teacher_ckpt)
if tok_t.pad_token is None:
    tok_t.pad_token = tok_t.eos_token

tok_s = AutoTokenizer.from_pretrained(student_ckpt)
if tok_s.pad_token is None:
    tok_s.pad_token = tok_s.eos_token
tok = tok_s
"""
    else:
        return """# 2. Locate Phase 1 Model Checkpoint
model_ckpt_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'config.json' in files:
        model_ckpt_dir = root
        break
print('Model Checkpoint Path:', model_ckpt_dir, os.listdir(model_ckpt_dir))

tok = AutoTokenizer.from_pretrained(model_ckpt_dir)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
"""

def get_common_cell4():
    return """# 3. Context Window Samples & Dataset Construction
with open('/kaggle/working/data_p3/label_vocab.json') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)
print('Number of classification classes:', num_labels)

with open('/kaggle/working/data_p3/CL_train.json') as f: cl_train = json.load(f)
with open('/kaggle/working/data_p3/CL_dev.json') as f: cl_dev = json.load(f)
with open('/kaggle/working/data_p3/IT_train.json') as f: it_train = json.load(f)
with open('/kaggle/working/data_p3/IT_dev.json') as f: it_dev = json.load(f)

# Train on 100% of labeled data (CL_train + CL_dev + IT_train + IT_dev = 18,987 sentences)
train_docs = cl_train + cl_dev + it_train + it_dev
dev_docs = cl_dev + it_dev

def build_context_samples(doc_list, is_test=False):
    samples = []
    if isinstance(doc_list, dict):
        items = [{'id': k, 'text': v} for k, v in doc_list.items()]
    else:
        items = doc_list
        
    for doc in items:
        doc_id = doc['id']
        sents = doc['text']
        labels = doc.get('labels', [])
        n = len(sents)
        for i, s in enumerate(sents):
            pct = int((i / max(1, n - 1)) * 100)
            prev_s = sents[i-1] if i > 0 else "NONE"
            next_s = sents[i+1] if i < n - 1 else "NONE"
            
            prompt = f"[DOC_POS: {pct}%] [PREV] {prev_s} [CURR] {s} [NEXT] {next_s}"
            sample = {
                'doc_id': doc_id,
                'sent_idx': i,
                'text': prompt
            }
            if not is_test and labels:
                sample['label'] = label2id[labels[i]]
            samples.append(sample)
    return samples

train_samples = build_context_samples(train_docs)
dev_samples = build_context_samples(dev_docs)
print(f'Train samples: {len(train_samples)}, Dev samples: {len(dev_samples)}')

class RRDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=192, is_test=False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        res = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'doc_id': item['doc_id'],
            'sent_idx': item['sent_idx']
        }
        if not self.is_test:
            res['label'] = torch.tensor(item['label'], dtype=torch.long)
        return res
"""

def get_model_cell5(exp):
    exp_id = exp['id']
    if exp_id == 'exp1_fullft':
        return """# 4. Initialize Model 1 with Full Fine-Tuning
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt_dir,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.config.pad_token_id = tok.pad_token_id
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Full Fine-Tuning: Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
"""
    elif exp_id == 'exp2_lora':
        return """# 4. Initialize Model 2 with Native PyTorch LoRA (Zero Dependencies)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt_dir,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.config.pad_token_id = tok.pad_token_id

class LoRALinear(nn.Module):
    def __init__(self, linear, r=16, alpha=32):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    def forward(self, x):
        return self.linear(x) + torch.matmul(x, torch.matmul(self.lora_B, self.lora_A).T) * self.scaling

# Freeze base model parameters
for p in model.parameters():
    p.requires_grad = False

# Attach LoRA adapters to attention projection layers
lora_count = 0
for name, module in model.named_modules():
    for child_name, child in module.named_children():
        if child_name in ['q_proj', 'v_proj', 'k_proj', 'o_proj'] and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, r=16, alpha=32))
            lora_count += 1

# Ensure sequence classification head is trainable
for name, p in model.named_parameters():
    if 'score' in name or 'classifier' in name:
        p.requires_grad = True

model = model.to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'LoRA attached to {lora_count} projection layers!')
print(f'Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
"""
    elif exp_id == 'exp3_lorma':
        return """# 4. Initialize Model 2 with Native PyTorch LORMA (Low-Rank Memory/Adapter)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt_dir,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.config.pad_token_id = tok.pad_token_id

class LORMALinear(nn.Module):
    def __init__(self, linear, r=16, alpha=32):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        # Low-rank memory adapter with learned layer-norm scaling
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.memory_gate = nn.Parameter(torch.ones(1) * 0.1)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    def forward(self, x):
        adapter_out = torch.matmul(x, torch.matmul(self.lora_B, self.lora_A).T) * self.scaling
        return self.linear(x) + self.memory_gate * adapter_out

for p in model.parameters():
    p.requires_grad = False

lorma_count = 0
for name, module in model.named_modules():
    for child_name, child in module.named_children():
        if child_name in ['q_proj', 'v_proj', 'k_proj', 'o_proj'] and isinstance(child, nn.Linear):
            setattr(module, child_name, LORMALinear(child, r=16, alpha=32))
            lorma_count += 1

for name, p in model.named_parameters():
    if 'score' in name or 'classifier' in name:
        p.requires_grad = True

model = model.to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'LORMA attached to {lorma_count} projection layers!')
print(f'Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
"""
    elif exp_id == 'exp4_layerwise':
        return """# 4. Initialize Model 3 with Layer-Wise Fine-Tuning (Freezing Bottom 8 Layers)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt_dir,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.config.pad_token_id = tok.pad_token_id

# Freeze bottom 8 transformer layers and embeddings, fine-tune top 8 layers + classification head
for name, p in model.named_parameters():
    is_top_layer = False
    for l_num in [8, 9, 10, 11, 12, 13, 14, 15]:
        if f'.{l_num}.' in name or f'layers.{l_num}' in name:
            is_top_layer = True
            break
    if is_top_layer or 'score' in name or 'classifier' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

model = model.to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Layer-wise adaptation configured! Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
"""
    elif exp_id == 'exp5_bitfit':
        return """# 4. Initialize Model 4 with BitFit (Only Biases + Classification Head Trainable)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt_dir,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.config.pad_token_id = tok.pad_token_id

trainable_params = 0
all_params = 0
for name, param in model.named_parameters():
    all_params += param.numel()
    if 'bias' in name or 'classifier' in name or 'score' in name:
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False

model = model.to(device)
print(f'BitFit applied! Trainable params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)')
"""
    elif exp_id == 'exp6_kd':
        return """# 4. Initialize Teacher (Model 2 RoPE LLaMA - Frozen) and Student (Model 1 GPT-2 - Trainable)
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_ckpt,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

student_model = AutoModelForSequenceClassification.from_pretrained(
    student_ckpt,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)
student_model.train()
student_model.config.pad_token_id = tok_s.pad_token_id

model = student_model
t_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
s_params = sum(p.numel() for p in student_model.parameters()) / 1e6
print(f'Teacher Model (Frozen): {t_params:.2f}M parameters')
print(f'Student Model (Trainable): {s_params:.2f}M parameters ({100*s_params/t_params:.1f}% size of teacher)')
"""

def get_training_cell6(exp):
    exp_id = exp['id']
    lr = exp['lr']
    epochs = exp['epochs']
    
    if exp_id == 'exp6_kd':
        return f"""# 5. Knowledge Distillation Training Loop (CE + KL-Divergence)
train_loader = DataLoader(RRDataset(train_samples, tok, max_length=192), batch_size=32, shuffle=True)
optimizer = AdamW(student_model.parameters(), lr={lr}, weight_decay=0.01)
epochs = {epochs}
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

{CLASS_WEIGHTS_CODE}
criterion = nn.CrossEntropyLoss(weight=class_weights)

TEMPERATURE = 3.0
ALPHA = 0.7

print(f'Starting Knowledge Distillation for {{epochs}} epochs ({{total_steps}} steps) on all 18,987 sentences...')
for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    student_model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Student forward
        s_logits = student_model(input_ids=input_ids, attention_mask=attn_mask).logits
        loss_ce = criterion(s_logits, labels)
        
        # Teacher forward
        with torch.no_grad():
            t_logits = teacher_model(input_ids=input_ids, attention_mask=attn_mask).logits
            
        # KL distillation loss
        loss_kd = F.kl_div(
            F.log_softmax(s_logits / TEMPERATURE, dim=-1),
            F.softmax(t_logits / TEMPERATURE, dim=-1),
            reduction='batchmean'
        ) * (TEMPERATURE ** 2)
        
        loss = ALPHA * loss_ce + (1.0 - ALPHA) * loss_kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f'Epoch {{epoch+1}}/{{epochs}} | Loss: {{total_loss/len(train_loader):.4f}} | CE: {{loss_ce.item():.4f}} | KD: {{loss_kd.item():.4f}} | Time: {{time.time()-t0:.1f}}s')
"""
    else:
        return f"""# 5. Training Loop with Class-Weighted Cross-Entropy
train_loader = DataLoader(RRDataset(train_samples, tok, max_length=192), batch_size=32, shuffle=True)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr={lr}, weight_decay=0.01)
epochs = {epochs}
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

{CLASS_WEIGHTS_CODE}
criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f'Starting {exp["method"]} training for {{epochs}} epochs ({{total_steps}} steps) on all 18,987 sentences...')
model.train()
for epoch in range(epochs):
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f'Epoch {{epoch+1}}/{{epochs}} | Loss: {{total_loss/len(train_loader):.4f}} | Time: {{time.time()-t0:.1f}}s')
"""

def get_eval_cell7(exp):
    zip_name = exp['zip_name']
    pred_name = exp['pred_name']
    title = exp['title']
    return f"""# 6. Validation and Test Inference
# 1. Validation on Dev Set
val_loader = DataLoader(RRDataset(dev_samples, tok, max_length=192), batch_size=32, shuffle=False)
model.eval()
all_preds = []
all_trues = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(labels.cpu().numpy())

val_macro_f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)
val_macro_p = precision_score(all_trues, all_preds, average='macro', zero_division=0)
val_macro_r = recall_score(all_trues, all_preds, average='macro', zero_division=0)
print(f'>>> Validation Strict Macro-F1: {{val_macro_f1:.4f}} | Prec: {{val_macro_p:.4f}} | Rec: {{val_macro_r:.4f}} <<<')

for lbl, i in sorted(label2id.items(), key=lambda x: x[1]):
    c_f1 = f1_score([1 if t==i else 0 for t in all_trues], [1 if p==i else 0 for p in all_preds], zero_division=0)
    print(f'  Class {{i:2d}} ({{lbl}}): F1 = {{c_f1:.4f}}')

# 2. Test Set Inference (Exact 2,197 sentences across 10 documents)
with open('/kaggle/working/data_p3/CL_test.json') as f: cl_test = json.load(f)
with open('/kaggle/working/data_p3/IT_test.json') as f: it_test = json.load(f)

test_samples_cl = build_context_samples(cl_test, is_test=True)
test_samples_it = build_context_samples(it_test, is_test=True)
all_test_samples = test_samples_cl + test_samples_it
print(f'Total test samples: {{len(all_test_samples)}} (Expected: 2197)')

test_loader = DataLoader(RRDataset(all_test_samples, tok, max_length=192, is_test=True), batch_size=32, shuffle=False)
test_doc_preds = {{d['id']: [] for d in [{{'id': k}} for k in list(cl_test.keys()) + list(it_test.keys())]}}

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        doc_ids = batch['doc_id']
        sent_indices = batch['sent_idx'].numpy()
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        
        for did, s_idx, p in zip(doc_ids, sent_indices, preds):
            test_doc_preds[did].append((int(s_idx), id2label[int(p)]))

# Sort by sentence index and extract ordered label list
final_predictions = {{}}
for did, sent_list in test_doc_preds.items():
    sent_list.sort(key=lambda x: x[0])
    final_predictions[did] = [lbl for _, lbl in sent_list]

# Verify exact sentence count per test document
for did, sents in final_predictions.items():
    expected_len = len(cl_test[did]) if did in cl_test else len(it_test[did])
    assert len(sents) == expected_len, f'Length mismatch in {{did}}: {{len(sents)}} vs expected {{expected_len}}'
print('✅ Exact length verification passed for all 10 test documents!')

# Package into CodaBench submission ZIP
pred_path = '/kaggle/working/prediction.json'
zip_path = '/kaggle/working/{zip_name}'

with open(pred_path, 'w', encoding='utf-8') as f:
    json.dump(final_predictions, f, ensure_ascii=False, indent=2)

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(pred_path, arcname='prediction.json')

print(f'Created {{zip_path}} ({{os.path.getsize(zip_path)/1024:.1f}} KB)')

# Upload to Hugging Face
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=zip_path,
    path_in_repo='phase3/{zip_name}',
    repo_id=TARGET_REPO,
    repo_type='dataset'
)
api.upload_file(
    path_or_fileobj=pred_path,
    path_in_repo='phase3/{pred_name}',
    repo_id=TARGET_REPO,
    repo_type='dataset'
)
print('Successfully uploaded Phase 3 submission to Hugging Face!')
"""


experiments = [
    {
        'id': 'exp1_fullft',
        'num': 1,
        'title': 'CS-781 Phase 3: Exp 1 — Full Fine-Tuning (Model 1 GPT-2)',
        'folder': 'kaggle_phase3_exp1',
        'code_file': 'cs781-phase3-exp1-fullft.ipynb',
        'account': 'ujjwalprakashgoodgod',
        'kernel_id': 'ujjwalprakashgoodgod/cs781-phase3-exp1-fullft',
        'dataset': 'ujjwalprakashgoodgod/cs781-phase1-m1-ckpt2000',
        'arch': 'gpt2',
        'method': 'Full Fine-Tuning',
        'lr': '3e-5',
        'epochs': 4,
        'zip_name': 'submission_phase3_exp1_fullft.zip',
        'pred_name': 'prediction_exp1_fullft.json',
    },
    {
        'id': 'exp2_lora',
        'num': 2,
        'title': 'CS-781 Phase 3: Exp 2 — LoRA Adaptation (Model 2 RoPE LLaMA)',
        'folder': 'kaggle_phase3_exp2',
        'code_file': 'cs781-phase3-exp2-lora.ipynb',
        'account': 'ujjwalprakashgoodgod',
        'kernel_id': 'ujjwalprakashgoodgod/cs781-phase3-exp2-lora',
        'dataset': 'ujjwalprakashgoodgod/cs781-phase1-m2-ckpt2500',
        'arch': 'llama',
        'method': 'LoRA (r=16, alpha=32)',
        'lr': '3e-4',
        'epochs': 5,
        'zip_name': 'submission_phase3_exp2_lora.zip',
        'pred_name': 'prediction_exp2_lora.json',
    },
    {
        'id': 'exp3_lorma',
        'num': 3,
        'title': 'CS-781 Phase 3: Exp 3 — LORMA Adaptation (Model 2 RoPE LLaMA)',
        'folder': 'kaggle_phase3_exp3',
        'code_file': 'cs781-phase3-exp3-lorma.ipynb',
        'account': 'ujjwalprakashgoodgod',
        'kernel_id': 'ujjwalprakashgoodgod/cs781-phase3-exp3-lorma',
        'dataset': 'ujjwalprakashgoodgod/cs781-phase1-m2-ckpt2500',
        'arch': 'llama',
        'method': 'LORMA (r=16, alpha=32)',
        'lr': '3e-4',
        'epochs': 5,
        'zip_name': 'submission_phase3_exp3_lorma.zip',
        'pred_name': 'prediction_exp3_lorma.json',
    },
    {
        'id': 'exp4_layerwise',
        'num': 4,
        'title': 'CS-781 Phase 3: Exp 4 — Layer-Wise Fine-Tuning (Model 3 GPT-NeoX)',
        'folder': 'kaggle_phase3_exp4',
        'code_file': 'cs781-phase3-exp4-layerwise.ipynb',
        'account': 'ujjwalprakashts',
        'kernel_id': 'ujjwalprakashts/cs781-phase3-exp4-layerwise',
        'dataset': 'ujjwalprakashts/cs781-phase1-m3-ckpt2500',
        'arch': 'gpt_neox',
        'method': 'Layer-Wise Fine-Tuning',
        'lr': '2e-4',
        'epochs': 5,
        'zip_name': 'submission_phase3_exp4_layerwise.zip',
        'pred_name': 'prediction_exp4_layerwise.json',
    },
    {
        'id': 'exp5_bitfit',
        'num': 5,
        'title': 'CS-781 Phase 3: Exp 5 — BitFit Bias Tuning (Model 4 Deep-Thin 16L)',
        'folder': 'kaggle_phase3_exp5',
        'code_file': 'cs781-phase3-exp5-bitfit.ipynb',
        'account': 'ujjwalprakashts',
        'kernel_id': 'ujjwalprakashts/cs781-phase3-exp5-bitfit',
        'dataset': 'ujjwalprakashts/cs781-phase1-m4-ckpt2500',
        'arch': 'deep_thin',
        'method': 'BitFit Bias Tuning',
        'lr': '8e-4',
        'epochs': 6,
        'zip_name': 'submission_phase3_exp5_bitfit.zip',
        'pred_name': 'prediction_exp5_bitfit.json',
    },
    {
        'id': 'exp6_kd',
        'num': 6,
        'title': 'CS-781 Phase 3: Exp 6 — Knowledge Distillation (Teacher M2 -> Student M1)',
        'folder': 'kaggle_phase3_exp6',
        'code_file': 'cs781-phase3-exp6-kd.ipynb',
        'account': 'ujjwalprakashgoodgod',
        'kernel_id': 'ujjwalprakashgoodgod/cs781-phase3-exp6-kd',
        'dataset': 'ujjwalprakashgoodgod/cs781-phase1-m2-ckpt2500',
        'dataset2': 'ujjwalprakashgoodgod/cs781-phase1-m1-ckpt2000',
        'arch': 'kd',
        'method': 'Knowledge Distillation',
        'lr': '3e-5',
        'epochs': 5,
        'zip_name': 'submission_phase3_exp6_kd.zip',
        'pred_name': 'prediction_exp6_kd.json',
    },
]

for exp in experiments:
    folder = f"{base_dir}/{exp['folder']}"
    os.makedirs(folder, exist_ok=True)
    
    # 1. Generate Notebook JSON
    nb = {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [f"# {exp['title']}\nMethodologically distinct Phase 3 Rhetorical Role Prediction submission using {exp['method']}."]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [f"print('Using Kaggle native PyTorch environment for {exp['method']}.')\n"]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [get_common_cell2()]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [get_common_cell3(exp['id'])]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [get_common_cell4()]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [get_model_cell5(exp)]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [get_training_cell6(exp)]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [get_eval_cell7(exp)]
            }
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10.12'}
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    nb_path = f"{folder}/{exp['code_file']}"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    # 2. Generate kernel-metadata.json
    dataset_sources = [exp['dataset']]
    if 'dataset2' in exp:
        dataset_sources.append(exp['dataset2'])
        
    meta = {
        'id': exp['kernel_id'],
        'title': exp['kernel_id'].split('/')[-1],
        'code_file': exp['code_file'],
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': dataset_sources
    }
    
    meta_path = f"{folder}/kernel-metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
        
    print(f"Generated {exp['id']:<15} -> {nb_path}")

print("\n🎉 ALL 6 PHASE 3 NOTEBOOKS AND METADATA SUCCESSFULLY GENERATED!")
