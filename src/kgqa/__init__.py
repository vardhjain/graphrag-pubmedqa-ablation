"""Knowledge Graph Question Answering — fair GraphRAG vs PlainRAG comparison.

A 4-arm ablation on PubMedQA that isolates exactly what a knowledge graph
contributes to retrieval-augmented QA, holding every other layer constant
(corpus, chunking, embedder, reranker, prompt, LLM, top-k).

Arms:
    plain           vector search -> top-k chunks (baseline)
    plain_rr        vector search -> cross-encoder rerank -> top-k chunks
    graph           plain_rr -> parent-paper expansion (full abstracts)
    graph_concepts  graph -> MeSH concept-hop expansion (related papers)
"""

__version__ = "1.0.0"
