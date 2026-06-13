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
confusion matrices and per-class F1. No LLM or database required — it only reads
`results/`, so it's light and deploys anywhere.

```bash
pip install -r app/requirements.txt    # light: streamlit + pandas + scikit-learn
streamlit run app/dashboard.py
```

### Deploy to Streamlit Community Cloud (free, always-on)

The dashboard is the project's hosted demo. `app/requirements.txt` sits next to
the entrypoint so Streamlit Cloud installs only the light deps (it searches the
entrypoint's directory before the heavy root `requirements.txt`).

1. Push these to `main`: `app/dashboard.py`, `app/requirements.txt`,
   `.streamlit/config.toml`, and the `results/` artifacts.
2. Go to <https://share.streamlit.io>, sign in with GitHub, authorize the repo.
3. **Create app → Deploy a public app from GitHub.**
4. Repository `vardhjain/Knowledge_Graph_Question_Answering`, Branch `main`,
   **Main file path `app/dashboard.py`**.
5. Advanced settings → Python 3.11; set the subdomain to `kgqa-ablation`
   (matches the README badge → `https://kgqa-ablation.streamlit.app`).
6. **Deploy.** Copy the final URL; if you changed the subdomain, update the
   badge + "Live demo" link in the root README.

> Tip: commit the per-sample `results/{arm}_results.json` files too (if you still
> have them from the benchmark run) to light up the confusion-matrix section.
