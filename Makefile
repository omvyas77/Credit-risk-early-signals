.PHONY: help download data features train scorer deep-input deep-baseline evaluate explain export writeup test lint all

CRES = uv run python -m cres

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 16

download: ## Raw ZIPs into $CRES_DATA_DIR/raw/, checksums into reports/data_manifest.json
	$(CRES) download

data: ## Unzip, join the archive to the live file, build the label
	$(CRES) ingest
	$(CRES) label

features: ## Windows and feature matrices into data/features/
	$(CRES) features

train: ## Classical models and the in-browser scorer's model
	$(CRES) train
	$(CRES) scorer

scorer: ## The in-browser scorer's model alone (also run by train)
	$(CRES) scorer

deep-input: ## Slim input for the GPU models on Colab (notebooks/README.md)
	$(CRES) deep-input

deep-baseline: ## TF-IDF LR on DistilBERT's training subsample
	$(CRES) deep --model lr_sub

evaluate: ## Every prediction file into reports/metrics.json and reports/summary.md
	$(CRES) evaluate

explain: ## Coefficients, SHAP, cross-product transfer and drift into reports/
	$(CRES) explain
	$(CRES) robustness

export: ## Site data into docs/data/ and the scorer bundle into docs/scorer/
	$(CRES) export

writeup: ## Fill the generated numbers in README.md and MODEL_CARD.md
	$(CRES) writeup

test: ## pytest, including the scorer parity check under Node
	uv run pytest

lint: ## ruff check and format check
	uv run ruff check .
	uv run ruff format --check .

all: download data features train evaluate explain export writeup ## Every stage in order
