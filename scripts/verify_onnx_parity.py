"""Prove the ONNX encoder is a safe swap for the torch one (hosted path).

The hosted agent runs ``KGQA_ENCODER=onnx`` (see render.yaml) to fit Render's
512MB tier, while the chunk vectors it searches were encoded by the *torch*
model at ingest time (``scripts/ingest_neo4j.py``). That only works if both
encoders put text in the same vector space. This script pins that empirically:

  1. cosine(torch_i, onnx_i) > 0.999 for varied texts (incl. a >256-token one,
     which must truncate identically rather than diverge or crash).
  2. Retrieval equivalence -- the thing that actually matters: with a corpus
     encoded by torch (simulating what's in Neo4j), an ONNX-encoded query must
     select the same top-3 chunks in the same order as a torch-encoded query.

Needs both stacks installed (dev machine, not CI -- CI never installs torch or
onnxruntime). Run before deploying any encoder change:

    python scripts/verify_onnx_parity.py

Exits nonzero on failure, so it can gate a deploy.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from kgqa.models import load_encoder, load_onnx_encoder  # noqa: E402
from kgqa.retrieval import ChunkStore  # noqa: E402

COSINE_FLOOR = 0.999

# Varied on purpose: a natural question, a terse phrase, punctuation/casing,
# unicode, and (last) a passage well past the model's 256-token limit so the
# truncation paths are compared, not just the easy cases.
TEXTS = [
    "Do preoperative statins reduce postoperative atrial fibrillation?",
    "aspirin",
    "Is vitamin D deficiency associated with increased mortality?",
    "COVID-19 vaccination and myocarditis risk in adolescents",
    "p < 0.0001; 95% CI [1.2, 3.4] -- statistically significant?",
    "Does laparoscopic surgery reduce hospital stay versus open surgery?",
    "Ácido acetilsalicílico y riesgo cardiovascular en pacientes diabéticos",
    "METFORMIN Versus Insulin In Gestational Diabetes: A RANDOMIZED Trial",
    "chunk of text with\nnewlines\tand\ttabs embedded in it",
    "randomized controlled trial of anticoagulation therapy " * 60,  # >256 tokens
]

# Stands in for the Neo4j-stored corpus in the retrieval check.
CORPUS = [
    "Preoperative statin therapy was associated with a lower incidence of postoperative atrial fibrillation after cardiac surgery.",
    "Statins reduce LDL cholesterol levels significantly in patients with hyperlipidemia.",
    "Vitamin D deficiency was associated with increased all-cause mortality in a large cohort study.",
    "Serum 25-hydroxyvitamin D concentrations correlate with bone mineral density in older adults.",
    "Laparoscopic cholecystectomy resulted in shorter hospital stays compared with open cholecystectomy.",
    "Open surgical repair remains the standard for complex abdominal aortic aneurysms.",
    "Aspirin use was associated with a reduced risk of colorectal cancer over ten years of follow-up.",
    "Low-dose aspirin did not significantly reduce cardiovascular events in primary prevention.",
    "Metformin was non-inferior to insulin for glycemic control in gestational diabetes.",
    "Insulin therapy remains first-line treatment for type 1 diabetes mellitus.",
    "Anticoagulation with warfarin increased bleeding risk in elderly patients with atrial fibrillation.",
    "Direct oral anticoagulants showed comparable efficacy to warfarin with fewer intracranial hemorrhages.",
    "COVID-19 mRNA vaccination was associated with a small increased risk of myocarditis in young males.",
    "Influenza vaccination reduced hospitalization rates among adults over 65 years of age.",
    "Physical exercise improved quality of life scores in patients with chronic heart failure.",
    "Dietary sodium restriction lowered blood pressure in hypertensive adults.",
    "Smoking cessation reduced the risk of lung cancer within five years of quitting.",
    "Screening colonoscopy detected adenomas in 25 percent of average-risk adults.",
    "Antibiotic prophylaxis did not reduce surgical site infections in clean procedures.",
    "Probiotic supplementation showed no significant effect on antibiotic-associated diarrhea.",
]

QUERIES = [
    "Do preoperative statins reduce postoperative atrial fibrillation?",
    "Is vitamin D deficiency associated with increased mortality?",
    "Does laparoscopic surgery reduce hospital stay versus open surgery?",
    "Does aspirin prevent colorectal cancer?",
    "Is metformin safe in gestational diabetes?",
]

TOP_K = 3


def main() -> int:
    print("Loading encoders (torch + onnx)...")
    torch_encoder = load_encoder()
    onnx_encoder = load_onnx_encoder()
    failures: list[str] = []

    # --- 1. per-text cosine parity -------------------------------------------
    print(f"\n[1/2] Cosine parity over {len(TEXTS)} texts (floor {COSINE_FLOOR}):")
    st_vecs = np.asarray(torch_encoder.encode(TEXTS, normalize_embeddings=True))
    onnx_vecs = np.asarray(onnx_encoder.encode(TEXTS, normalize_embeddings=True))

    if st_vecs.shape != onnx_vecs.shape:
        print(f"  FAIL shape mismatch: torch {st_vecs.shape} vs onnx {onnx_vecs.shape}")
        return 1

    for i, text in enumerate(TEXTS):
        cos = float(np.dot(st_vecs[i], onnx_vecs[i]))
        label = (text[:57] + "...") if len(text) > 60 else text
        status = "ok  " if cos > COSINE_FLOOR else "FAIL"
        if cos <= COSINE_FLOOR:
            failures.append(f"cosine {cos:.6f} for {label!r}")
        print(f"  {status} cos={cos:.6f}  {label!r}")

    # --- 2. retrieval equivalence against a torch-encoded corpus --------------
    print(f"\n[2/2] Top-{TOP_K} retrieval order over a torch-encoded corpus "
          f"({len(CORPUS)} chunks, {len(QUERIES)} queries):")
    corpus_vecs = np.asarray(torch_encoder.encode(CORPUS, normalize_embeddings=True))
    store = ChunkStore(
        ids=[f"Chunks/{i}_0" for i in range(len(CORPUS))],
        paper_keys=[str(i) for i in range(len(CORPUS))],
        texts=list(CORPUS),
        embeddings=corpus_vecs,
    )

    for query in QUERIES:
        st_hits = store.search(torch_encoder.encode([query], normalize_embeddings=True)[0], TOP_K)
        onnx_hits = store.search(onnx_encoder.encode([query], normalize_embeddings=True)[0], TOP_K)
        label = (query[:57] + "...") if len(query) > 60 else query
        if list(st_hits) == list(onnx_hits):
            print(f"  ok   {list(st_hits)}  {label!r}")
        else:
            failures.append(f"top-{TOP_K} order {list(onnx_hits)} != {list(st_hits)} for {label!r}")
            print(f"  FAIL torch={list(st_hits)} onnx={list(onnx_hits)}  {label!r}")

    # --- verdict --------------------------------------------------------------
    print()
    if failures:
        print(f"FAILED ({len(failures)} check(s)) -- do NOT deploy KGQA_ENCODER=onnx:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: ONNX encoder matches the torch encoder's vector space and "
          "retrieval order. Safe to deploy KGQA_ENCODER=onnx.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
