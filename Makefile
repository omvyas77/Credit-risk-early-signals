.PHONY: download data features train evaluate explain export test lint all

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
