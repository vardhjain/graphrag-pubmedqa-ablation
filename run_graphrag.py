"""
GraphRAG benchmark runner — standalone script (no Jupyter required).
Runs the same pipeline as GraphRAG.ipynb.

Set ARANGO_PASS before running:
    Windows: $env:ARANGO_PASS = "your_password"
    Linux:   export ARANGO_PASS=your_password
"""

import os
import sys
import time
import subprocess

# Add repo root to path so shared_utils is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Start Ollama (idempotent) ─────────────────────────────────────────────────
print('[Ollama] Starting server...')
subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(3)

# ── Imports ───────────────────────────────────────────────────────────────────
import pickle
import requests
import numpy as np
from arango import ArangoClient
from arango.exceptions import ServerConnectionError, ArangoServerError
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from shared_utils import (
    EMBEDDING_MODEL, LLM_MODEL, TOP_K_FINAL,
    BENCHMARK_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT,
    FuzzyEvaluator, Evaluator, call_ollama,
)

# ── Config ────────────────────────────────────────────────────────────────────
ARANGO_HOST = os.environ.get('ARANGO_HOST', 'https://bfc25a0e3c74.arangodb.cloud:8529')
ARANGO_USER = os.environ.get('ARANGO_USER', 'root')
ARANGO_PASS = os.environ.get('ARANGO_PASS', '')
ARANGO_DB   = os.environ.get('ARANGO_DB',   'pubmed_graph')

TOP_K_CANDIDATES  = 75
CROSS_ENCODER     = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
VECTOR_CACHE_FILE = os.path.join(ROOT, 'pubmed_vectors_cache.pkl')
RESULTS_DIR       = os.path.join(ROOT, 'results')
RESULTS_FILE      = os.path.join(RESULTS_DIR, 'graphrag_results.json')
BENCHMARK_N       = 100   # must match run_plainrag.py

os.makedirs(RESULTS_DIR, exist_ok=True)

if not ARANGO_PASS:
    raise EnvironmentError(
        'Set ARANGO_PASS before running:\n'
        '    PowerShell : $env:ARANGO_PASS = "your_password"\n'
        '    CMD        : set ARANGO_PASS=your_password'
    )

# ── ArangoDB ──────────────────────────────────────────────────────────────────

def connect_arango(host, user, password, db_name, max_retries=5):
    client = ArangoClient(hosts=host)
    for attempt in range(max_retries):
        try:
            sys_db = client.db('_system', username=user, password=password)
            sys_db.version()
            db = client.db(db_name, username=user, password=password)
            print('[ArangoDB] Connected.')
            return db
        except (ServerConnectionError, ArangoServerError) as exc:
            wait = (attempt + 1) * 5
            print(f'[ArangoDB] Attempt {attempt + 1} failed. Retrying in {wait}s...')
            time.sleep(wait)
    raise ConnectionError('Could not connect to ArangoDB.')


def load_chunk_vectors(db, collection='Chunks', cache_file=VECTOR_CACHE_FILE):
    if os.path.exists(cache_file):
        print(f'[Cache] Loading from {cache_file}...')
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            ids, texts, embeddings = data['ids'], data['texts'], data['embeddings']
            if len(embeddings) > 0:
                print(f'[Cache] Loaded {len(embeddings):,} vectors.')
                return ids, texts, np.array(embeddings)
        except Exception as exc:
            print(f'[Cache] Corrupted ({exc}). Re-downloading...')

    print('[Index] Downloading vectors from ArangoDB (first run only)...')
    ids, texts, embeddings = [], [], []
    BATCH, offset = 5000, 0

    try:
        total = list(db.aql.execute(f'RETURN LENGTH({collection})'))[0]
    except Exception:
        total = 0

    with tqdm(total=total, desc='Downloading') as pbar:
        while True:
            aql = f'''
                FOR c IN {collection}
                    FILTER c.embedding != null
                    LIMIT {offset}, {BATCH}
                    RETURN {{ id: c._id, text: c.text, emb: c.embedding }}
            '''
            try:
                batch = list(db.aql.execute(aql, ttl=3600))
            except Exception as exc:
                print(f'[Index] Error: {exc}')
                if '503' in str(exc):
                    time.sleep(5)
                break

            if not batch:
                break
            for doc in batch:
                ids.append(doc['id'])
                texts.append(doc['text'])
                embeddings.append(doc['emb'])

            pbar.update(len(batch))
            offset += len(batch)
            if len(batch) < BATCH:
                break
            time.sleep(0.1)

    embeddings_np = np.array(embeddings)
    if ids:
        with open(cache_file, 'wb') as f:
            pickle.dump({'ids': ids, 'texts': texts, 'embeddings': embeddings_np}, f)
        print(f'[Cache] Saved {len(ids):,} vectors.')
    return ids, texts, embeddings_np

# ── GraphRAG ──────────────────────────────────────────────────────────────────

class GraphRAG:
    def __init__(self, db, chunk_ids, chunk_texts, chunk_embeddings):
        self.db               = db
        self.chunk_ids        = chunk_ids
        self.chunk_texts      = chunk_texts
        self.chunk_embeddings = chunk_embeddings
        print('[Model] Loading Sentence Transformer...')
        self.encoder  = SentenceTransformer(EMBEDDING_MODEL)
        print('[Model] Loading CrossEncoder...')
        self.reranker = CrossEncoder(CROSS_ENCODER)
        print('[GraphRAG] Initialised.')

    def retrieve(self, query: str) -> str:
        if len(self.chunk_embeddings) == 0:
            return 'No context available.'

        query_emb  = self.encoder.encode([query])
        sims       = cosine_similarity(query_emb, self.chunk_embeddings)[0]
        top_idx    = np.argsort(sims)[-TOP_K_CANDIDATES:][::-1]
        candidates = [(self.chunk_texts[i], self.chunk_ids[i]) for i in top_idx]

        pairs    = [[query, text] for text, _ in candidates]
        scores   = self.reranker.predict(pairs)
        best_idx = np.argsort(scores)[::-1][:TOP_K_FINAL]
        best_ids = [candidates[i][1] for i in best_idx]

        return self._expand_via_graph(best_ids)

    def _expand_via_graph(self, chunk_ids: list) -> str:
        aql = '''
            WITH Papers, Chunks
            FOR cid IN @ids
                LET chunk = DOCUMENT(cid)
                FOR paper IN 1..1 INBOUND chunk HAS_CONTEXT
                    LET all_chunks = (
                        FOR c IN 1..1 OUTBOUND paper HAS_CONTEXT
                            RETURN c.text
                    )
                    LET full_abstract = CONCAT_SEPARATOR(" ", all_chunks)
                    RETURN { title: paper.title, abstract: full_abstract }
        '''
        try:
            rows  = list(self.db.aql.execute(aql, bind_vars={'ids': chunk_ids}))
            seen  = set()
            parts = []
            for row in rows:
                title = row.get('title', 'Unknown Study')
                if title in seen:
                    continue
                seen.add(title)
                abstract = row.get('abstract', '')
                parts.append('=== STUDY: ' + title + ' ===\n' + abstract)
            return '\n\n'.join(parts) if parts else 'No context found.'
        except Exception as exc:
            print(f'[GraphRAG] Graph expansion failed ({exc}). Using raw chunks.')
            return '\n\n'.join(
                'Excerpt: ' + text
                for text, _ in candidates[:TOP_K_FINAL]
            )

    def answer_benchmark(self, question: str) -> str:
        context = self.retrieve(question)
        prompt  = 'Context:\n' + context + '\n\nQuestion: ' + question
        return call_ollama(prompt, system=BENCHMARK_SYSTEM_PROMPT, temperature=0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    db = connect_arango(ARANGO_HOST, ARANGO_USER, ARANGO_PASS, ARANGO_DB)
    chunk_ids, chunk_texts, chunk_embeddings = load_chunk_vectors(db)
    rag = GraphRAG(db, chunk_ids, chunk_texts, chunk_embeddings)

    dataset   = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    fuzzy     = FuzzyEvaluator()
    evaluator = Evaluator('GraphRAG')

    print(f'\n=== GraphRAG Benchmark  (n={BENCHMARK_N}) ===')

    for i, item in enumerate(dataset):
        if i >= BENCHMARK_N:
            break

        question = item['question']
        gt       = item['final_decision']

        t0      = time.time()
        raw     = rag.answer_benchmark(question)
        latency = time.time() - t0

        pred = fuzzy.extract_answer(raw)
        evaluator.record(gt, pred, latency)

        icon = 'v' if pred == gt else 'x'
        print(f'[{i+1:3d}]  GT={gt:<5}  Pred={pred:<5}  {icon}  ({latency:.1f}s)')

    evaluator.report()
    evaluator.save(RESULTS_FILE)
