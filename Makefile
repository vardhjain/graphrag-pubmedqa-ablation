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

format:  ## Auto-fix lint issues and format with ruff
	ruff check --fix src scripts tests app backend
	ruff format src scripts tests app backend

ingest:  ## Build the ArangoDB knowledge graph (needs ARANGO_PASS)
	python scripts/ingest.py

# The published Colab run used num_ctx=8192 (config.py's default is 4096; the
# env var overrides it without touching config). Each result file's meta block
# records the num_ctx actually used.
benchmark:  ## Run all five arms: 4-arm ladder + graph_norr (needs ARANGO_PASS + Ollama)
	@export LLM_NUM_CTX=8192; \
	for arm in plain plain_rr graph graph_concepts graph_norr; do \
		echo "===== $$arm ====="; \
		python scripts/run_benchmark.py --arm $$arm --n 200; \
	done

compare:  ## Aggregate results into table, McNemar tests, and figure
	python scripts/compare.py

# Optional sixth arm is deliberately NOT included in the historical benchmark
# or the published Holm family. Separate output directory avoids clobbering.
.PHONY: check-parent-parity benchmark-parent-control compare-parent-control
check-parent-parity:  ## Check full-abstract parity via Arango; NO LLM
	python scripts/check_parent_parity.py --n 100 --seed 42

benchmark-parent-control:  ## Optional fresh paired graph/flat runs; local Ollama
	@export LLM_NUM_CTX=8192; \
	python scripts/check_parent_parity.py --n 100 --seed 42 && \
	for arm in graph flat_parent; do \
		echo "===== $arm (fresh parent control) ====="; \
		python scripts/run_benchmark.py --arm $arm --n 200 --output results/parent_control/${arm}_results.json || exit 1; \
	done; \
	python scripts/compare_parent_baseline.py \
		--flat results/parent_control/flat_parent_results.json \
		--graph results/parent_control/graph_results.json \
		--output results/parent_control/comparison.json

compare-parent-control:  ## Compare fresh saved runs; no inference
	python scripts/compare_parent_baseline.py \
		--flat results/parent_control/flat_parent_results.json \
		--graph results/parent_control/graph_results.json

chat:  ## Launch the Gradio chat demo (needs ArangoDB + Ollama)
	python app/chat_app.py

dashboard:  ## Launch the Streamlit results dashboard
	streamlit run app/dashboard.py

clean:  ## Remove caches and generated vector cache
	rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info \
		pubmed_vectors_cache.pkl
	find . -type d -name __pycache__ -exec rm -rf {} +
