"""
Shared utilities for the Knowledge Graph QA project.

GraphRAG.ipynb and Plain_RAG/Plain_RAG.ipynb both import from this module
to guarantee identical embedding models, LLM, prompts, answer extraction,
and evaluation — so the only variables between them are the retrieval strategy
and context assembly method.
"""

import re
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Model identifiers (both notebooks must use these exact values) ────────────
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'   # 384-dim
LLM_MODEL       = 'deepseek-r1:8b'
OLLAMA_API      = 'http://localhost:11434/api/chat'
TOP_K_FINAL     = 3   # documents fed to the LLM in both pipelines

# ── Prompts (word-for-word identical in both pipelines) ───────────────────────
BENCHMARK_SYSTEM_PROMPT = (
    'You are a PubMedQA annotator. Classify the answer as yes, no, or maybe.\n\n'
    'Guidelines:\n'
    '- YES  : the study finds a positive outcome, correlation, or association,\n'
    '         even if further research is recommended.\n'
    '- NO   : the study finds no significant difference or a negative result.\n'
    '- MAYBE: only if the abstract explicitly states inconclusive results\n'
    '         with no supporting data.\n\n'
    'End your response with exactly: Final Answer: [yes/no/maybe]'
)

CHAT_SYSTEM_PROMPT = (
    'You are a helpful medical AI assistant. '
    'Use the provided research abstracts to answer the user question. '
    'Cite specific study titles when making claims. '
    'If studies conflict, explain the conflict. '
    'If the context is insufficient, say so and give your best assessment.'
)


# ── Answer extraction ─────────────────────────────────────────────────────────

class FuzzyEvaluator:
    """Extracts and normalises yes/no/maybe from verbose model output."""

    def extract_answer(self, text: str) -> str:
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).lower()
        match = re.search(r'final answer\s*:\s*(yes|no|maybe)', clean)
        if match:
            return match.group(1)
        matches = re.findall(r'\b(yes|no|maybe)\b', clean)
        return matches[-1] if matches else 'maybe'


# ── Evaluation ────────────────────────────────────────────────────────────────

class Evaluator:
    """Records predictions and generates a full evaluation report."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.y_true:    list = []
        self.y_pred:    list = []
        self.latencies: list = []

    def record(self, ground_truth: str, prediction: str, latency: float = 0.0):
        pred = prediction.lower().strip()
        if pred not in ('yes', 'no', 'maybe'):
            pred = 'maybe'
        self.y_true.append(ground_truth.lower().strip())
        self.y_pred.append(pred)
        self.latencies.append(latency)

    def report(self) -> dict:
        if not self.y_true:
            print('No data recorded.')
            return {}

        labels  = ['yes', 'no', 'maybe']
        acc     = accuracy_score(self.y_true, self.y_pred)
        total_t = sum(self.latencies)
        avg_t   = total_t / len(self.latencies) if self.latencies else 0.0

        print(f"\n{'=' * 52}")
        print(f"  {self.model_name} — Evaluation Report")
        print(f"{'=' * 52}")
        print(f"  Samples      : {len(self.y_true)}")
        print(f"  Accuracy     : {acc:.2%}")
        print(f"  Total time   : {total_t:.1f}s  |  Avg/query : {avg_t:.1f}s")
        print(f"{'-' * 52}")
        print(classification_report(self.y_true, self.y_pred, labels=labels, zero_division=0))

        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix — {self.model_name}')
        plt.tight_layout()
        plt.show()

        return {
            'model':       self.model_name,
            'accuracy':    acc,
            'samples':     len(self.y_true),
            'total_time':  total_t,
            'avg_latency': avg_t,
            'y_true':      self.y_true,
            'y_pred':      self.y_pred,
        }

    def save(self, path: str):
        data = {
            'model':       self.model_name,
            'accuracy':    accuracy_score(self.y_true, self.y_pred) if self.y_true else 0,
            'samples':     len(self.y_true),
            'total_time':  sum(self.latencies),
            'avg_latency': sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            'y_true':      self.y_true,
            'y_pred':      self.y_pred,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Results saved to {path}')


# ── LLM interface ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, system: str = '',
                temperature: float = 0.0,
                model: str = LLM_MODEL) -> str:
    """Single synchronous call to the local Ollama API."""
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    payload = {
        'model':    model,
        'messages': messages,
        'stream':   False,
        'options':  {'temperature': temperature, 'num_ctx': 4096},
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()['message']['content']
