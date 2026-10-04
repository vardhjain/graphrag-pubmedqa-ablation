---
---

[**▶ Live demo**](https://graphrag-pubmedqa-ablation.vercel.app) &nbsp;·&nbsp; [**Results dashboard**](https://graphrag-pubmedapp-ablation-busqcyjfc7kywhq9m9flwo.streamlit.app) &nbsp;·&nbsp; [**GitHub repo**](https://github.com/vardhjain/graphrag-pubmedqa-ablation) &nbsp;·&nbsp; [**Project report (PDF)**](https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/docs/Project_Report.pdf) &nbsp;·&nbsp; [**Slides**](https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/docs/Graph_RAG_PPT.pptx)

## What this is

AI assistants answer questions more reliably when they first look up relevant
documents and read them before answering (*retrieval-augmented generation*, or
**RAG**). A popular upgrade, **GraphRAG**, stores the documents in a *knowledge
graph* that records how pieces of text relate, and uses those links to find
better context.

Many demos claim GraphRAG beats plain RAG, but the comparisons are often unfair:
the graph version quietly gets extra help, or the correct answer leaks into what
the AI reads. This project removes those advantages and asks: **when everything
else is held equal, which part of the graph actually helps?**

We tested four setups on 200 yes/no/maybe questions from
[PubMedQA](https://pubmedqa.github.io/), a dataset of questions about medical
research papers. Each setup adds exactly one technique to the one before it, so
every change in accuracy can be credited to a single cause.

![Benchmark pipeline and the four setups](https://raw.githubusercontent.com/vardhjain/graphrag-pubmedqa-ablation/main/assets/architecture.svg)

## Results

| Setup | What it adds | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| `plain` | Basic RAG: look up the 3 closest text fragments | 30.0% | 29.7% |
| `plain_rr` | + a second model re-ranks the fragments | 37.0% | 35.2% |
| **`graph`** | **+ follow the graph to each fragment's full abstract** | **59.5%** | **50.5%** |
| `graph_concepts` | + also pull in papers sharing medical topics | 57.5% | 50.0% |

![Accuracy of the four setups](https://raw.githubusercontent.com/vardhjain/graphrag-pubmedqa-ablation/main/results/ablation.png)

**What we found:**

- **Giving the AI the full abstract instead of one fragment was the decisive
  win:** +22.5 percentage points, statistically significant (p < 0.0001).
- **The re-ranker helped a little, but not conclusively** (+7 points, p = 0.08).
- **Pulling in related papers did not help** (−2 points, p = 0.69) and made
  answers about five times slower.

The graph helps by *deepening* context (more of the right document), not by
*broadening* it (more documents). We expected the opposite and report what the
data showed. Method details and caveats are in
[RESULTS.md](https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/RESULTS.md).

## Explore

- **[Live demo](https://graphrag-pubmedqa-ablation.vercel.app)** — ask a biomedical question and see the graph path behind the answer. It is a smaller version of the experiment (1,000 papers, no re-ranker), so treat it as an illustration; the first answer can take 1–2 minutes while the free-tier server wakes up.
- **[Results dashboard](https://graphrag-pubmedapp-ablation-busqcyjfc7kywhq9m9flwo.streamlit.app)** — interactive charts of the numbers above
- **[Source code & README](https://github.com/vardhjain/graphrag-pubmedqa-ablation)** — code, tests, and how to run it yourself
- **[Project report (PDF)](https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/docs/Project_Report.pdf)** and **[slides](https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/docs/Graph_RAG_PPT.pptx)**
