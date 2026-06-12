"""Prompts — word-for-word identical across every arm.

The benchmark prompt classifies a PubMedQA question as yes/no/maybe. It is the
same string for PlainRAG and GraphRAG; only the retrieved ``context`` differs.
"""

BENCHMARK_SYSTEM_PROMPT = (
    "You are a PubMedQA annotator. Classify the answer as yes, no, or maybe.\n\n"
    "Guidelines:\n"
    "- YES  : the study finds a positive outcome, correlation, or association,\n"
    "         even if further research is recommended.\n"
    "- NO   : the study finds no significant difference or a negative result.\n"
    "- MAYBE: only if the abstract explicitly states inconclusive results\n"
    "         with no supporting data.\n\n"
    "End your response with exactly: Final Answer: [yes/no/maybe]"
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful medical AI assistant. "
    "Use the provided research abstracts to answer the user question. "
    "If studies conflict, explain the conflict. "
    "If the context is insufficient, say so and give your best assessment."
)


def build_prompt(context: str, question: str) -> str:
    """Assemble the user-turn prompt — identical structure for every arm."""
    return f"Context:\n{context}\n\nQuestion: {question}"
