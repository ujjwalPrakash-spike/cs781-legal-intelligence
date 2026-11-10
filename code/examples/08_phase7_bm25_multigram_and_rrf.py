#!/usr/bin/env python3
"""
08_phase7_bm25_multigram_and_rrf.py
-------------------------------------
CS781 Capstone - Viva Voce Demonstration Script
Phase 7: Multi-Gram BM25 Retrieval & Reciprocal Rank Fusion (RRF)

Why this matters for Viva:
1. Legal Prior Case Retrieval (PCR) requires capturing exact legal statutes and doctrine names.
2. Standard unigram BM25 dilutes compound phrases like "anticipatory bail", "Section 302 IPC",
   and "habeas corpus" into generic words ("bail", "section", "corpus").
   Multi-gram BM25 indexes both unigrams and bigrams.
3. Dense embeddings capture broad semantic similarity, while BM25 captures exact phrase citations.
4. Reciprocal Rank Fusion (RRF) merges rankings using rank position:
       RRF(d) = sum_{m in rankers} 1 / (k + rank_m(d))   with k=60
   This avoids score-scale mismatch between unbounded BM25 scores and bounded [-1, 1] cosine similarities.
"""

import math
import re
from collections import Counter, defaultdict

def tokenize_and_multigrams(text):
    """
    Tokenizes text and extracts unigrams and bigrams.
    """
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = clean_text.split()
    unigrams = words
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return unigrams + bigrams


class MultiGramBM25:
    """
    BM25 Okapi implementation with multi-gram support.
    """
    def __init__(self, corpus_dict, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(corpus_dict.keys())
        self.doc_lengths = {}
        self.doc_term_freqs = {}
        self.doc_freqs = defaultdict(int)
        self.total_docs = len(self.doc_ids)
        
        # Build index
        total_len = 0
        for doc_id, text in corpus_dict.items():
            tokens = tokenize_and_multigrams(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)
            
            tf = Counter(tokens)
            self.doc_term_freqs[doc_id] = tf
            for term in tf.keys():
                self.doc_freqs[term] += 1
                
        self.avg_doc_len = total_len / self.total_docs if self.total_docs > 0 else 1.0

    def get_scores(self, query):
        q_tokens = tokenize_and_multigrams(query)
        scores = {}
        
        for doc_id in self.doc_ids:
            doc_len = self.doc_lengths[doc_id]
            tf_dict = self.doc_term_freqs[doc_id]
            score = 0.0
            
            for term in q_tokens:
                if term not in self.doc_freqs:
                    continue
                df = self.doc_freqs[term]
                # Standard Robertson-Spärck Jones IDF
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                tf = tf_dict.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
                
            scores[doc_id] = score
            
        # Return ranked doc list
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def reciprocal_rank_fusion(sparse_ranked, dense_ranked, k=60):
    """
    Merges sparse (BM25) and dense rankings using Reciprocal Rank Fusion (RRF).
    RRF(d) = sum_{ranker} 1 / (k + rank)
    """
    rrf_scores = defaultdict(float)
    
    # Process sparse ranking
    for rank, (doc_id, score) in enumerate(sparse_ranked, start=1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        
    # Process dense ranking
    for rank, (doc_id, score) in enumerate(dense_ranked, start=1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def main():
    print("=== Phase 7: Multi-Gram BM25 Retrieval & Reciprocal Rank Fusion ===")
    
    corpus = {
        "Case_101": "The appellant was arrested for homicide and conspiracy under Section 302 IPC. The trial judge rejected anticipatory bail.",
        "Case_102": "Civil property dispute concerning partition of ancestral land under Hindu Succession Act. No criminal charges.",
        "Case_103": "Application for anticipatory bail under Section 438 CrPC in relation to allegations under Section 302 IPC and criminal conspiracy.",
        "Case_104": "Company petition under Companies Act regarding shareholder voting rights and insolvency resolution process.",
        "Case_105": "Criminal appeal regarding circumstantial evidence in murder case under Section 302 IPC. Anticipatory bail previously dismissed."
    }
    
    query = "anticipatory bail under Section 302 IPC criminal conspiracy"
    print(f"Query: \"{query}\"\n")
    
    # 1. Multi-gram BM25 Retrieval
    bm25 = MultiGramBM25(corpus)
    sparse_ranked = bm25.get_scores(query)
    
    print("Multi-Gram BM25 Rankings (Sparse):")
    for rank, (doc_id, score) in enumerate(sparse_ranked, start=1):
        print(f"  Rank {rank}: {doc_id} (BM25 Score: {score:.4f}) -> \"{corpus[doc_id][:60]}...\"")
        
    # 2. Simulated Dense Semantic Retrieval (Bi-encoder Cosine Similarity)
    # Notice: Dense might rank Case_105 higher due to global semantic vocabulary
    dense_ranked = [
        ("Case_103", 0.88),
        ("Case_105", 0.84),
        ("Case_101", 0.79),
        ("Case_102", 0.32),
        ("Case_104", 0.15)
    ]
    
    print("\nDense Bi-Encoder Rankings (Semantic):")
    for rank, (doc_id, sim) in enumerate(dense_ranked, start=1):
        print(f"  Rank {rank}: {doc_id} (Cosine Sim: {sim:.2f})")
        
    # 3. Reciprocal Rank Fusion (k=60)
    final_rrf = reciprocal_rank_fusion(sparse_ranked, dense_ranked, k=60)
    
    print("\nReciprocal Rank Fusion (RRF k=60) Hybrid Rankings:")
    for rank, (doc_id, rrf_score) in enumerate(final_rrf, start=1):
        sparse_pos = next(i for i, (d, _) in enumerate(sparse_ranked, 1) if d == doc_id)
        dense_pos = next(i for i, (d, _) in enumerate(dense_ranked, 1) if d == doc_id)
        print(f"  Rank {rank}: {doc_id} (RRF Score: {rrf_score:.5f}) "
              f"[Sparse Rank: {sparse_pos}, Dense Rank: {dense_pos}]")
        
    print(f"\nTop retrieved document: {final_rrf[0][0]} perfectly matches compound phrases:")
    print(f"  \"{corpus[final_rrf[0][0]]}\"")
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
