# Standalone Viva Voce Coding Examples
**Course:** CS781: Deep Learning for Natural Language Processing (IIT Kanpur)  
**Student:** Ujjwal Prakash (Roll No: 241111)

These 8 Python scripts are self-contained demonstrations of the core mathematical and architectural algorithms across the 7 phases of the capstone project. They run standalone without external Kaggle datasets, printing intermediate tensor shapes, loss terms, and metric values.

---

## Quick Execution

You can run each script directly with Python (PyTorch $\ge 2.0$ required):

```bash
# Using the local environment
/home/ujjwal/kaggle-env/bin/python reports/learning_guide/code_examples/01_rope_attention_from_scratch.py
```

Or run all 8 in sequence:

```bash
for script in reports/learning_guide/code_examples/*.py; do
    echo "=========================================="
    echo "Running $(basename "$script")"
    /home/ujjwal/kaggle-env/bin/python "$script"
done
```

---

## Script Inventory

| Script | Phase | Key Algorithm / Mechanism | Viva Talking Point |
| :--- | :--- | :--- | :--- |
| `01_rope_attention_from_scratch.py` | Phase 1 | Rotary Position Embeddings (RoPE) & Causal Scaled Dot-Product Attention | 2D orthogonal rotation preserves relative token distances $(m-n)$ without additive position embeddings. |
| `02_native_peft_adapters.py` | Phase 2–5 | Native PyTorch `LoRALinear` & `LORMALinear` | Low-rank factorization $\Delta W = \frac{\alpha}{r} A B$ cuts trainable parameters by 99.3% (49K vs 7.1M). |
| `03_knowledge_distillation_step.py` | Phase 1–5 | Knowledge Distillation Step with $T^2$ scaling | Temperature scaling softens logits; $T^2$ multiplier balances soft KL gradients against hard cross-entropy. |
| `04_phase2_offset_alignment_demo.py` | Phase 2 | Byte-Pair Token Offset Alignment for Legal NER | Resolves recurring statute names where naive `str.find()` erroneously collapses duplicate mentions. |
| `05_phase4_head_tail_chunking.py` | Phase 4 | Head-Tail Chunking & Binary Threshold Calibration | Indian court rulings average 3,400+ tokens; combining Head + Tail retains charges and orders; threshold sweep optimizes F1. |
| `06_phase5_coordinate_ascent_thresholds.py` | Phase 5 | 100-Class Coordinate Ascent for Multi-Label Statutes | Non-differentiable Macro-F1 cannot use gradient descent; 1D coordinate sweeps prevent 0-prediction failures for rare statutes. |
| `07_phase6_mbr_medoid_consensus.py` | Phase 6 | Minimum Bayes Risk (MBR) Consensus Decoding | Unstructured text sequences cannot use majority voting; pairwise ROUGE-L identifies central consensus and rejects hallucinations. |
| `08_phase7_bm25_multigram_and_rrf.py` | Phase 7 | Multi-Gram BM25 + Dense Retrieval + RRF ($k=60$) | Multi-grams preserve legal compounds ("Section 302 IPC"); RRF merges ranks by position, avoiding score scale distortion. |
