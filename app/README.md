# Apps

Two optional front-ends. Install their deps with `pip install -r requirements-app.txt`.

## `chat_app.py` — live GraphRAG chat (Gradio)

An interactive assistant over the winning `graph` arm: it retrieves from the
knowledge graph, answers with `deepseek-r1:8b`, and cites the source PubMed IDs.

```bash
python app/chat_app.py            # http://localhost:7860
python app/chat_app.py --share    # public share link (handy on Colab)
python app/chat_app.py --concepts # use the graph_concepts arm
```

This is a **live** demo, so it needs the backend running: a reachable ArangoDB
(`ARANGO_HOST` / `ARANGO_PASS`) and Ollama with `deepseek-r1:8b` pulled. To host
it on **Hugging Face Spaces**, set the Space SDK to Gradio and `app_file:
app/chat_app.py`, and point `ARANGO_HOST`/`ARANGO_PASS` at a hosted database via
Space secrets.

## `dashboard.py` — results dashboard (Streamlit)

Visualizes the saved benchmark: per-arm accuracy/F1, the paired McNemar tests,
the ablation figure, and (if the per-sample `results/*_results.json` are present)
confusion matrices and per-class F1.

```bash
streamlit run app/dashboard.py
```

No LLM or database required — it only reads `results/`, so it deploys to
**Streamlit Cloud** as a click-to-view link straight from the repo.
