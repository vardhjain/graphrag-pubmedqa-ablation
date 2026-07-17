.DEFAULT_GOAL := help
.PHONY: help install install-dev install-app test lint format ingest benchmark compare chat dashboard clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install dev dependencies (tests + lint)
	pip install -r requirements-dev.txt

install-app:  ## Install UI dependencies (gradio + streamlit)
	pip install -r requirements-app.txt

test:  ## Run the test suite
	pytest

lint:  ## Lint with ruff
	ruff check src scripts tests app backend

format:  ## Auto-fix lint issues with ruff
	ruff check --fix src scripts tests app backend

ingest:  ## Build the ArangoDB knowledge graph (needs ARANGO_PASS)
	python scripts/ingest.py

benchmark:  ## Run all four arms (needs ARANGO_PASS + Ollama)
	@for arm in plain plain_rr graph graph_concepts; do \
		echo "===== $$arm ====="; \
		python scripts/run_benchmark.py --arm $$arm --n 200; \
	done

compare:  ## Aggregate results into table, McNemar tests, and figure
	python scripts/compare.py

chat:  ## Launch the Gradio chat demo (needs ArangoDB + Ollama)
	python app/chat_app.py

dashboard:  ## Launch the Streamlit results dashboard
	streamlit run app/dashboard.py

clean:  ## Remove caches and generated vector cache
	rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info \
		pubmed_vectors_cache.pkl
	find . -type d -name __pycache__ -exec rm -rf {} +
