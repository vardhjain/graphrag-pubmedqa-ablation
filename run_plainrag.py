"""
Plain RAG benchmark runner — standalone script (no Jupyter required).
Runs the same pipeline as Plain_RAG/Plain_RAG.ipynb.
"""

import os
import sys
import time
import subprocess

# Add repo root to path so shared_utils is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Start Ollama (idempotent — safe to call if already running) ───────────────
print('[Ollama] Starting server...')
subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(3)

# ── Imports (after Ollama is up) ─────────────────────────────────────────────
import os
import re
import pickle
import requests
import numpy as np
import faiss
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from shared_utils import (
    EMBEDDING_MODEL, LLM_MODEL, TOP_K_FINAL,
    BENCHMARK_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT,
    FuzzyEvaluator, Evaluator, call_ollama,
)

# ── Config ────────────────────────────────────────────────────────────────────
PLAIN_RAG_DIR = os.path.join(ROOT, 'Plain_RAG')
INDEX_FILE    = os.path.join(PLAIN_RAG_DIR, 'pubmed_rag_index.bin')
DATA_FILE     = os.path.join(PLAIN_RAG_DIR, 'pubmed_rag_data.pkl')
RESULTS_DIR   = os.path.join(ROOT, 'results')
RESULTS_FILE  = os.path.join(RESULTS_DIR, 'plainrag_results.json')
BENCHMARK_N   = 100   # change to run more samples

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── PlainRAG ──────────────────────────────────────────────────────────────────

class PlainRAG:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[PlainRAG] Initialising on {self.device}')
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        self.dim      = self.embedder.get_sentence_embedding_dimension()
        self.index    = None
        self.documents    = []
        self.labeled_data = []

    def load_data(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
            print('[Index] Loading from disk...')
            self.index = faiss.read_index(INDEX_FILE)
            with open(DATA_FILE, 'rb') as f:
                saved = pickle.load(f)
            self.documents    = saved['documents']
            self.labeled_data = saved['labeled_data']
            print(f'[Index] Loaded {len(self.documents):,} documents.')
            return

        print('[Data] Building index from scratch (first run — takes ~10 min)...')
        ds_labeled    = load_dataset('qiaojin/PubMedQA', 'pqa_labeled',    split='train')
        ds_unlabeled  = load_dataset('qiaojin/PubMedQA', 'pqa_unlabeled',  split='train')
        ds_artificial = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')

        def process_split(ds, name):
            docs = []
            for item in tqdm(ds, desc=f'Processing {name}'):
                docs.append({
                    'text':           ' '.join(item['context']['contexts']),
                    'pubid':          item['pubid'],
                    'question':       item.get('question', ''),
                    'final_decision': item.get('final_decision'),
                })
            return docs

        self.labeled_data = process_split(ds_labeled, 'labeled')
        all_docs = list(self.labeled_data)
        all_docs.extend(process_split(ds_unlabeled,  'unlabeled'))
        all_docs.extend(process_split(ds_artificial, 'artificial'))
        self.documents = all_docs

        print(f'[Embed] Encoding {len(self.documents):,} documents...')
        embeddings = self.embedder.encode(
            [d['text'] for d in self.documents],
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print('[Index] Building FAISS IndexFlatIP...')
        index_cpu = faiss.IndexFlatIP(self.dim)
        index_cpu.add(embeddings)
        os.makedirs(PLAIN_RAG_DIR, exist_ok=True)
        faiss.write_index(index_cpu, INDEX_FILE)
        with open(DATA_FILE, 'wb') as f:
            pickle.dump({'documents': self.documents, 'labeled_data': self.labeled_data}, f)
        print(f'[Index] Saved to {INDEX_FILE}.')
        self.index = index_cpu

    def retrieve(self, query: str) -> str:
        vec        = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = self.index.search(vec, TOP_K_FINAL)
        retrieved  = [self.documents[i] for i in indices[0] if i != -1]
        return '\n\n'.join(
            'Abstract ' + str(i + 1) + ': ' + doc['text']
            for i, doc in enumerate(retrieved)
        )

    def answer_benchmark(self, question: str) -> str:
        context = self.retrieve(question)
        prompt  = 'Context:\n' + context + '\n\nQuestion: ' + question
        return call_ollama(prompt, system=BENCHMARK_SYSTEM_PROMPT, temperature=0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rag = PlainRAG()
    rag.load_data()

    fuzzy     = FuzzyEvaluator()
    evaluator = Evaluator('Plain RAG')

    print(f'\n=== Plain RAG Benchmark  (n={BENCHMARK_N}) ===')

    for i, item in enumerate(rag.labeled_data):
        if i >= BENCHMARK_N:
            break

        question = item.get('question', '')
        gt       = item.get('final_decision', '')
        if not question or not gt:
            continue

        t0      = time.time()
        raw     = rag.answer_benchmark(question)
        latency = time.time() - t0

        pred = fuzzy.extract_answer(raw)
        evaluator.record(gt, pred, latency)

        icon = 'v' if pred == gt else 'x'
        print(f'[{i+1:3d}]  GT={gt:<5}  Pred={pred:<5}  {icon}  ({latency:.1f}s)')

    evaluator.report()
    evaluator.save(RESULTS_FILE)
