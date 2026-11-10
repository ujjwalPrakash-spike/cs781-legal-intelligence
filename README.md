# In-House Legal Intelligence System
### Pretrained Legal Language Models and Parameter-Efficient Domain Adaptation
**Course:** CS781: Large Language Models, Department of Computer Science & Engineering, IIT Kanpur  
**Author:** Ujjwal Prakash (Roll No: 241111, `ujjwal24@iitk.ac.in`)  
**Codabench Username:** `ujjwalprakashgoodgod` / `241111`  
**Official Report:** [report/CS781-CP-Ujjwal-Prakash-241111-report.pdf](report/CS781-CP-Ujjwal-Prakash-241111-report.pdf)  

---

## 1. Executive Summary & Problem Formulation

Indian judicial proceedings generate large volumes of unstructured case law across the Supreme Court of India, state High Courts, and appellate tribunals. While commercial cloud LLMs (such as GPT-4 or Claude) generate fluent text, deploying them in legal practice creates severe compliance and confidentiality barriers:
* **The Advocates Act (1961)** and **Bar Council of India** rules mandate absolute professional secrecy over client materials.
* **The Digital Personal Data Protection Act (DPDPA 2023)** restricts transmitting unredacted personal identifiers, non-public arbitration testimonies, and sensitive judicial records to external multi-tenant cloud APIs.
* **Competition Constraints:** Air-gapped on-premise execution, no external pretrained model weights, and a strict parameter ceiling ($\le 300\text{M}$ parameters).

In this capstone project, I designed, built, and evaluated a sovereign, self-hosted legal NLP pipeline from the ground up:
1. **Pretrained 4 foundation architectures from scratch** on 125,000 raw Indian legal documents (120M tokens) without external weights.
2. Selected the champion **RoPE LLaMA** foundation (124M parameters, 15.35 perplexity) and transferred it across six complex downstream legal tasks.
3. Implemented **6 adaptation regimes** in native PyTorch (Full Fine-Tuning, LoRA, LORMA, Layer-Wise Freezing, BitFit, and Knowledge Distillation).
4. Conducted **42 validated competition submissions** on the official Codabench benchmark across all seven phases, consistently meeting and exceeding the four-submission requirement.

---

## 2. Competition Results & Leaderboard Summary

| Phase | Task Description | Primary Metric | Dev Score | Codabench Official | Valid Submissions | Best Configuration |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | Legal LM Pretraining | Perplexity $\downarrow$ / Loss | **15.35** (PPL) | **2.731** (Loss) | 4 | Model 2: RoPE LLaMA + SwiGLU |
| **2** | Legal Named Entity Recognition | Strict Micro-F1 $\uparrow$ | **0.6842** | **0.6710** | 6 | Layer-Wise (Top-4) + UTF-8 Byte Offset Mapping |
| **3** | Rhetorical Role Prediction | Macro-F1 $\uparrow$ | **0.8124** | **0.7985** | 6 | Multi-Model Probability Ensemble (BiLSTM + 7.86x Weights) |
| **4** | Court Judgment Prediction | Macro-F1 $\uparrow$ | **0.6932** | **0.6815** | 6 | Layer-Wise + Head-Tail Chunking + Threshold ($\tau=0.325$) |
| **5** | Legal Statute Identification | Macro-F1 $\uparrow$ | **0.3338** | **0.3294** | 6 | Layer-Wise + Coordinate-Ascent Multi-Thresholds |
| **6** | Abstractive Legal Summarization | BERTScore F1 $\uparrow$ | **0.3840** (R-L) | **0.6780** (BERT) | 6 | 6-Model Minimum Bayes Risk (MBR) Medoid Consensus |
| **7** | Prior Case Precedent Retrieval | Micro-F1@7 $\uparrow$ | **0.4124** | **0.3936** | 8 | Hybrid Multigram BM25 + Dense Bi-Encoder (RRF $k=60$) |
| **Total** | **All 7 Phases Combined** | — | — | — | **42** | **Fully Verified on Codabench** |

---

## 3. Architecture & Foundation Pretraining (Phase 1)

Four transformer architectures were built from absolute zero and trained across identical tokenized blocks of 1,024 tokens on Kaggle Tesla T4 GPUs:

```
[Raw Legal Corpus: 125,000 Documents]
                 │
      [Byte-Pair BPE Tokenizer (50,257 vocab)]
                 │
┌────────────────┴────────────────┬───────────────────────────────┐
▼                                 ▼                               ▼
Model 1: GPT-2 Baseline     Model 2: RoPE LLaMA (Champion)   Model 3: GPT-NeoX
- Absolute Learned Pos.     - Rotary Embeddings (RoPE)       - Partial RoPE (25%)
- LayerNorm + GeLU          - RMSNorm + SwiGLU Gating        - LayerNorm + GeLU
- 124.44M params            - 124.05M params                 - 123.65M params
- Perplexity: 18.80         - Perplexity: 15.35 🏆           - Perplexity: 17.03
```

### Why Model 2 (RoPE LLaMA) Won Decisively:
* **Rotary Position Embeddings (RoPE):** Operates on 2D head coordinate rotations:
  $$\mathbf{q}_m^T \mathbf{k}_n = (\mathbf{R}_m \mathbf{W}_q \mathbf{x}_m)^T (\mathbf{R}_n \mathbf{W}_k \mathbf{x}_n) = \mathbf{x}_m^T \mathbf{W}_q^T \mathbf{R}_{n-m} \mathbf{W}_k \mathbf{x}_n$$
  Absolute token indices cancel out. Attention scores depend solely on relative clause distance $(n - m)$, allowing the model to naturally resolve complex multi-clause statutory qualifications.
* **RMSNorm:** Omits mean-centering from standard LayerNorm:
  $$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \mathbf{g}$$
  Saves GPU memory bandwidth and runs ~15% faster during training.
* **SwiGLU Gating:** Dynamic nonlinear filtering:
  $$\text{SwiGLU}(\mathbf{x}) = (\mathbf{x} \mathbf{W}_{\text{gate}} \cdot \text{SiLU}(\mathbf{x} \mathbf{W}_{\text{gate}})) \odot (\mathbf{x} \mathbf{W}_{\text{up}}) \mathbf{W}_{\text{down}}$$
* **Autoregressive Generation Details:** Implemented left-padding (`padding_side="left"`) to prevent `<pad>` token contamination in causal prompts, and enabled KV-caching (`use_cache=True`) for $\mathcal{O}(1)$ autoregressive step latency.

---

## 4. Parameter-Efficient Fine-Tuning (Native PyTorch PEFT)

All adaptation techniques were implemented directly in native PyTorch without external PEFT libraries:

### 1. Native LoRA (Low-Rank Adaptation)
Decomposes weight updates $\Delta \mathbf{W}$ into low-rank factorized matrices across attention projections ($Q, K, V, O$):
$$\mathbf{W}_{\text{adapted}} = \mathbf{W}_{\text{frozen}} + \frac{\alpha}{r} (\mathbf{B} \cdot \mathbf{A})$$
* Matrix $\mathbf{A} \in \mathbb{R}^{r \times d}$ is initialized with Gaussian noise $\sim \mathcal{N}(0, 1/r)$.
* Matrix $\mathbf{B} \in \mathbb{R}^{d \times r}$ is initialized to **exact zeros**.
* *Mathematical Guarantee:* At step 0, $\mathbf{B} \cdot \mathbf{A} = \mathbf{0}$, ensuring the pretrained foundation model begins in an undisturbed state while gradients immediately flow into both matrices on step 1.
* Trains only $1.5\%$ of parameters.

### 2. LORMA (Low-Rank Multi-Head Memory Adapter)
Introduces a learnable scalar memory gate $\boldsymbol{\mu} \in \mathbb{R}^1$:
$$\mathbf{W}_{\text{adapted}}(\mathbf{x}) = \mathbf{W}_{\text{frozen}}(\mathbf{x}) + \boldsymbol{\mu} \cdot \left[ \frac{\alpha}{r} \mathbf{x} (\mathbf{B} \cdot \mathbf{A})^T \right]$$
Allows individual attention heads to dynamically modulate the injection of task-specific memory versus foundational knowledge.

### 3. Layer-Wise Fine-Tuning (Top-4 Layers)
Freezes layers 0 through 7 (syntax, morphology, legal lexicon) and fine-tunes only layers 8 through 11 and the task classification head, applying layer-wise learning rate decay ($\gamma = 0.85$). In Phase 4, this prevented catastrophic forgetting, outperforming full fine-tuning by $+0.02$ Macro-F1.

### 4. BitFit (Bias-Only Fine-Tuning)
Freezes every 2D weight matrix in the transformer and trains **only the 1D bias parameters**:
$$\mathbf{y} = \mathbf{x} \mathbf{W}_{\text{frozen}} + (\mathbf{b}_{\text{frozen}} + \Delta \mathbf{b}_{\text{trainable}})$$
Updates $<0.15\%$ of parameters (~140,000 parameters), yet achieved $0.6314$ Macro-F1 in Phase 4 with over $90\%$ optimizer memory savings.

### 5. Knowledge Distillation
Transfers knowledge from Model 2 (Teacher) to Model 1 (Student) via softened probabilities:
$$\mathcal{L}_{\text{KD}} = (1 - \lambda) \mathcal{L}_{\text{CE}}(\mathbf{y}, \sigma(\mathbf{z}_s)) + \lambda T^2 D_{\text{KL}}\left( \sigma\left(\frac{\mathbf{z}_t}{T}\right) \parallel \sigma\left(\frac{\mathbf{z}_s}{T}\right) \right)$$
Using temperature $T = 3.0$ and $\lambda = 0.5$. The $T^2$ multiplier compensates for gradient shrinkage caused by $1/T$ logit scaling.

---

## 5. Phase-by-Phase Technical Breakdown

### Phase 2: Legal Named Entity Recognition (12 Classes)
* **Goal:** Extract 12 legal entities (Petitioner, Respondent, Court, Judge, Statute, Provision, etc.) with exact character boundaries.
* **The Offset Drift Bug & Fix:** Naive whitespace collapsing shifted character coordinates by 1–6 characters, resulting in 0.0 F1 on Codabench. We bypassed string normalization entirely, reading character spans directly from unnormalized UTF-8 bytes via tokenizer `return_offsets_mapping=True`.
* **Result:** Reached **0.6842 dev / 0.6710 Codabench** Strict Span Micro-F1.

### Phase 3: Rhetorical Role Labeling (13 Discourse Classes)
* **Goal:** Classify judicial sentences into discourse roles (Facts, Ruling Lower Court, Argument, Statute, Ratio, Present Court, Dissent).
* **Class Imbalance Strategy:** `Facts` had 8,420 sentences while `Dissent` had only 210. We applied inverse frequency weighting in Cross-Entropy, penalizing errors on `Dissent` by **7.86x**.
* **Discourse Modeling:** Added a lightweight BiLSTM layer over pooled sentence tokens to capture context flow across sentences.
* **Result:** Multi-model probability ensemble achieved **0.8124 dev / 0.7985 Codabench** Macro-F1.

### Phase 4: Court Judgment Prediction (Binary Classification)
* **Goal:** Predict whether an appeal will be **Allowed (1)** or **Dismissed (0)**.
* **Head-Tail Dual-Window Chunking:** Cases span 8,000+ tokens. We extracted the first 1,536 tokens (Facts + Issue) and the last 1,536 tokens (Ratio + Operative Order) and concatenated them into a single 3,072-token tensor, bypassing irrelevant middle witness depositions.
* **Threshold Calibration:** Appeals are dismissed ~70% of the time, causing standard 0.50 cutoff models to rarely predict "Allowed". A 41-point validation grid search discovered an optimal decision threshold of **$\tau^* = 0.325$**, jumping Macro-F1 from $0.612$ to **0.6932 dev / 0.6815 Codabench**.

### Phase 5: Legal Statute Identification (100 Statutory Provisions)
* **Goal:** Multi-label classification across 100 Indian statutory provisions (IPC, CrPC, Evidence Act).
* **Coordinate-Ascent Threshold Optimization:** Rare statutes rarely exceeded $0.18$ probability, scoring 0.0 under a 0.50 threshold. We optimized an independent threshold $\tau_c \in [0.05, 0.60]$ per class via coordinate ascent on validation F1, elevating Macro-F1 from $0.182$ to **0.3338 dev / 0.3294 Codabench**.

### Phase 6: Abstractive Legal Summarization
* **Goal:** Generate concise, legally accurate headnote summaries.
* **Minimum Bayes Risk (MBR) Medoid Consensus:** You cannot average text sentences across models. We generated candidate summaries from 6 fine-tuned checkpoints and computed a $6 \times 6$ pairwise ROUGE-L similarity matrix. The candidate with the highest sum of pairwise similarities (the statistical medoid) was selected, filtering out individual model hallucinations.
* **Result:** Achieved **0.6780 BERTScore F1** on Codabench.

### Phase 7: Prior Case Precedent Retrieval
* **Goal:** Retrieve the top 7 historical court precedents from an archive of 4,320 candidates.
* **Hybrid Fusion:**
  1. *Lexical BM25 with Multigrams:* Matches exact statutory citations ($k_1 = 1.5, b = 0.75$).
  2. *Dense Bi-Encoder:* Generates 768-d semantic embeddings from our Phase 1 model to match conceptual synonyms via cosine similarity.
* **Reciprocal Rank Fusion (RRF):**
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)} \quad (k = 60)$$
  Eliminates score scale distortion and selects the top 7 precedents, driving **Micro-F1@7 to 0.4124 dev / 0.3936 Codabench**.

---

## 6. Repository Structure

```
cs781-legal-intelligence/
├── README.md                                    # Comprehensive documentation
├── report/
│   └── CS781-CP-Ujjwal-Prakash-241111-report.pdf # Official 10-page final report
└── code/
    ├── phase1/                                  # Foundation Pretraining from scratch
    │   ├── cs781-phase1-model2-rope-llama.ipynb # Champion RoPE LLaMA training pipeline
    │   └── kernel-metadata.json
    ├── phase2/                                  # Legal NER & Offset Recovery
    │   ├── cs781-phase2-exp4-prefixtune.ipynb
    │   └── kernel-metadata.json
    ├── phase3/                                  # Rhetorical Role Labeling
    │   ├── generate_phase3_notebooks.py         # PEFT generator (LoRA, LORMA, BitFit, KD)
    │   ├── cs781-phase3-exp4-layerwise.ipynb
    │   └── kernel-metadata.json
    ├── phase4/                                  # Court Judgment Prediction
    │   ├── cs781-phase4-exp2-layerwise.ipynb    # Head-tail chunking pipeline
    │   ├── verify_and_ensemble_phase4.py        # Optimal threshold calibration (tau=0.325)
    │   └── kernel-metadata.json
    ├── phase5/                                  # Statute Identification
    │   ├── cs781-phase5-exp2-layerwise.ipynb
    │   ├── verify_and_ensemble_phase5.py        # 100-class coordinate ascent thresholding
    │   └── kernel-metadata.json
    ├── phase6/                                  # Legal Summarization
    │   ├── cs781-phase6-exp2v2-layerwise.ipynb
    │   ├── verify_and_ensemble_phase6.py        # MBR Medoid consensus selection
    │   └── kernel-metadata.json
    ├── phase7/                                  # Prior Case Retrieval
    │   ├── generate_phase7_notebooks.py         # Bi-Encoder trainer & BM25 indexing
    │   └── verify_and_ensemble_phase7.py        # Reciprocal Rank Fusion (k=60)
    └── examples/                                # Standalone Native PyTorch Demos
        ├── 01_rope_attention_from_scratch.py    # RoPE 2D rotation & self-attention
        ├── 02_native_peft_adapters.py           # Native LoRA, LORMA, and BitFit modules
        ├── 03_knowledge_distillation_step.py    # Softened KL-divergence step
        ├── 04_phase2_offset_alignment_demo.py   # Byte-pair offset mapping
        ├── 05_phase4_head_tail_chunking.py      # Dual-window long-document chunking
        ├── 06_phase5_coordinate_ascent_thresholds.py # Multi-label threshold search
        ├── 07_phase6_mbr_medoid_consensus.py    # ROUGE-L pairwise MBR consensus
        └── 08_phase7_bm25_multigram_and_rrf.py  # BM25 + Dense RRF ranking
```

---

## 7. How to Run & Reproduce

### 1. Environment Setup
```bash
git clone https://github.com/ujjwalPrakash-spike/cs781-legal-intelligence.git
cd cs781-legal-intelligence

# Recommended Python 3.10+ virtual environment
python3 -m venv env
source env/bin/activate
pip install torch transformers datasets accelerate scikit-learn rouge-score bert-score rank-bm25
```

### 2. Running Standalone Demonstrations
You can verify each architectural component locally without external datasets:
```bash
python3 code/examples/01_rope_attention_from_scratch.py
python3 code/examples/02_native_peft_adapters.py
python3 code/examples/03_knowledge_distillation_step.py
python3 code/examples/05_phase4_head_tail_chunking.py
python3 code/examples/06_phase5_coordinate_ascent_thresholds.py
python3 code/examples/07_phase6_mbr_medoid_consensus.py
python3 code/examples/08_phase7_bm25_multigram_and_rrf.py
```

### 3. Running Kaggle Phase Notebooks
All phase notebooks in `code/phase1/` through `code/phase7/` are self-contained Kaggle kernels configured with `kernel-metadata.json`. You can push and run them directly via the Kaggle CLI:
```bash
kaggle kernels push -p code/phase1/
kaggle kernels push -p code/phase4/
```

---

## 8. Citation & Academic Integrity

```bibtex
@misc{prakash2026legalintelligence,
  author = {Ujjwal Prakash},
  title = {In-House Legal Intelligence System: Pretrained Legal Language Models and Parameter-Efficient Domain Adaptation},
  year = {2026},
  institution = {Department of Computer Science and Engineering, IIT Kanpur},
  howpublished = {\url{https://github.com/ujjwalPrakash-spike/cs781-legal-intelligence}}
}
```
*All experiments adhered strictly to the CS781 academic integrity code and Codabench competition rules. No external pretrained model weights or commercial cloud APIs were utilized.*
