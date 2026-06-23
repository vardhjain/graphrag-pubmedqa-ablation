"""Gradio chat demo over the GraphRAG (`graph`) arm — the ablation's winner.

    python app/chat_app.py              # local: http://localhost:7860
    python app/chat_app.py --share      # public share link (Colab / remote)
    python app/chat_app.py --concepts   # use the graph_concepts arm instead

Requirements: `pip install gradio` (see requirements-app.txt), a reachable
ArangoDB (set ARANGO_HOST / ARANGO_PASS), and a running Ollama with the model
pulled. This is a *live* demo — it retrieves from the graph and calls the LLM.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

EXAMPLES = [
    "Do preoperative statins reduce postoperative atrial fibrillation?",
    "Is vitamin D deficiency associated with increased mortality?",
    "Does laparoscopic surgery reduce hospital stay versus open surgery?",
]


def _strip_think(text: str) -> str:
    """Drop the reasoning model's <think>...</think> block for a clean answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_retriever(use_concepts: bool):
    from kgqa.config import ArangoConfig
    from kgqa.models import connect_arango, load_encoder, load_reranker
    from kgqa.retrieval import ChunkStore, GraphRetriever

    db = connect_arango(ArangoConfig())
    cache = os.path.join(ROOT, "pubmed_vectors_cache.pkl")
    store = ChunkStore.from_arango(db, cache_file=cache)
    print(f"[demo] {len(store):,} chunks loaded")
    return GraphRetriever(store, load_encoder(), db,
                          reranker=load_reranker(), use_concepts=use_concepts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--share", action="store_true", help="create a public share link")
    parser.add_argument("--concepts", action="store_true", help="use the graph_concepts arm")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    import gradio as gr

    rag = build_retriever(args.concepts)

    def respond(message, history):
        result = rag.chat(message)
        answer = _strip_think(result["answer"]) or "_No answer produced._"
        sources = result.get("sources", [])
        if sources:
            links = "\n".join(
                f"- [PMID {pid}](https://pubmed.ncbi.nlm.nih.gov/{pid}/)" for pid in sources
            )
            answer += f"\n\n**Sources**\n{links}"
        return answer

    gr.ChatInterface(
        fn=respond,
        title="PubMed GraphRAG assistant",
        description=(
            "Graph-augmented retrieval over PubMedQA: matched chunks are expanded "
            "to full abstracts via the knowledge graph, then answered by "
            "deepseek-r1:8b. Answers cite the source PubMed IDs."
        ),
        examples=EXAMPLES,
    ).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
