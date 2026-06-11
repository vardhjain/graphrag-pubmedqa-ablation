"""
Comparison script — loads both result JSON files and prints/plots the comparison.
Run after run_plainrag.py and run_graphrag.py have both completed.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

ROOT         = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(ROOT, 'results')
GRAPHRAG_F   = os.path.join(RESULTS_DIR, 'graphrag_results.json')
PLAINRAG_F   = os.path.join(RESULTS_DIR, 'plainrag_results.json')
LABELS       = ['yes', 'no', 'maybe']

for path in [GRAPHRAG_F, PLAINRAG_F]:
    if not os.path.exists(path):
        print(f'Missing: {path}')
        print('Run run_graphrag.py and run_plainrag.py first.')
        sys.exit(1)

with open(GRAPHRAG_F) as f:
    gr = json.load(f)
with open(PLAINRAG_F) as f:
    pr = json.load(f)

# ── Summary table ─────────────────────────────────────────────────────────────
gr_f1 = f1_score(gr['y_true'], gr['y_pred'], labels=LABELS, average='macro', zero_division=0)
pr_f1 = f1_score(pr['y_true'], pr['y_pred'], labels=LABELS, average='macro', zero_division=0)

summary = pd.DataFrame({
    'Model':           [gr['model'],                     pr['model']],
    'Accuracy (%)':    [round(gr['accuracy'] * 100, 2),  round(pr['accuracy'] * 100, 2)],
    'Macro F1 (%)':    [round(gr_f1 * 100, 2),           round(pr_f1 * 100, 2)],
    'Avg Latency (s)': [round(gr['avg_latency'], 2),     round(pr['avg_latency'], 2)],
    'Samples':         [gr['samples'],                   pr['samples']],
})
print('\n' + '=' * 60)
print('  RESULTS SUMMARY')
print('=' * 60)
print(summary.to_string(index=False))
print('=' * 60)

# ── Accuracy / F1 / Latency bars ──────────────────────────────────────────────
models  = [gr['model'], pr['model']]
accs    = [gr['accuracy'] * 100, pr['accuracy'] * 100]
f1s     = [gr_f1 * 100, pr_f1 * 100]
lats    = [gr['avg_latency'], pr['avg_latency']]
colours = ['#2196F3', '#FF9800']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, values, label in zip(axes, [accs, f1s, lats], ['Accuracy (%)', 'Macro F1 (%)', 'Avg Latency (s)']):
    bars = ax.bar(models, values, color=colours, width=0.4, edgecolor='white')
    ax.set_title(label)
    ax.set_ylabel(label)
    if 'Latency' not in label:
        ax.set_ylim(0, 100)
    for bar, val in zip(bars, values):
        suffix = 's' if 'Latency' in label else ('%' if '%' in label else '')
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 1.03 if 'Latency' in label else val + 1.5,
                f'{val:.1f}{suffix}', ha='center', fontweight='bold')

plt.suptitle(
    f'GraphRAG vs Plain RAG — Head-to-Head  (n={gr["samples"]} samples)',
    fontsize=13, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison_bars.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(RESULTS_DIR, "comparison_bars.png")}')

# ── Confusion matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, res, cmap in zip(axes, [gr, pr], ['Blues', 'Oranges']):
    cm = confusion_matrix(res['y_true'], res['y_pred'], labels=LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(res['model'] + f'  (acc={res["accuracy"]:.2%})')

plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison_confusion.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(RESULTS_DIR, "comparison_confusion.png")}')

# ── Per-class F1 ─────────────────────────────────────────────────────────────
gr_f1s = f1_score(gr['y_true'], gr['y_pred'], labels=LABELS, average=None, zero_division=0)
pr_f1s = f1_score(pr['y_true'], pr['y_pred'], labels=LABELS, average=None, zero_division=0)

x, width = np.arange(len(LABELS)), 0.35
fig, ax  = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width / 2, gr_f1s * 100, width, label=gr['model'], color='#2196F3', edgecolor='white')
b2 = ax.bar(x + width / 2, pr_f1s * 100, width, label=pr['model'], color='#FF9800', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel('F1 Score (%)'); ax.set_ylim(0, 100)
ax.set_title('Per-class F1 Score'); ax.legend()
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison_f1.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(RESULTS_DIR, "comparison_f1.png")}')

# ── Verdict ───────────────────────────────────────────────────────────────────
acc_delta = (gr['accuracy'] - pr['accuracy']) * 100
f1_delta  = (gr_f1 - pr_f1) * 100
lat_delta = gr['avg_latency'] - pr['avg_latency']
winner    = gr['model'] if acc_delta >= 0 else pr['model']

print('\n' + '=' * 60)
print('  VERDICT')
print('=' * 60)
print(f'  Winner          : {winner}')
print(f'  Accuracy delta  : {acc_delta:+.2f} pp  (GraphRAG − Plain RAG)')
print(f'  Macro F1 delta  : {f1_delta:+.2f} pp')
print(f'  Latency delta   : {lat_delta:+.2f}s')
print('=' * 60)
