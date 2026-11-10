.PHONY: download data features train deep-input deep-baseline evaluate explain export test lint all

CRES = uv run python -m cres

download:
	$(CRES) download

data:
	$(CRES) ingest
	$(CRES) label

features:
	$(CRES) features

train:
	$(CRES) train

# Slim input for the deep models on Colab (notebooks/README.md), and the same-data baseline.
deep-input:
	$(CRES) deep-input

deep-baseline:
	$(CRES) deep --model lr_sub

evaluate:
	$(CRES) evaluate

explain:
	$(CRES) explain
	$(CRES) robustness

export:
	$(CRES) export

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

all: download data features train evaluate explain export
