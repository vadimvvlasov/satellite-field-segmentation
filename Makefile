.PHONY: sync train predict docker clean lint format

CONFIG ?= configs/train.yaml
INPUT  ?= data/sample_tile.tif
OUTPUT ?= output/fields.gpkg
IMAGE  ?= field-seg

# Install dependencies
sync:
	uv sync

sync-dev:
	uv sync --all-extras

# Lint & format
lint:
	uv run ruff check src/ predict.py

format:
	uv run ruff format src/ predict.py

# Training & inference
train:
	uv run python src/train.py --config $(CONFIG)

predict:
	mkdir -p $(shell dirname $(OUTPUT))
	uv run python predict.py --input $(INPUT) --output $(OUTPUT)

# Docker
docker:
	docker build -t $(IMAGE) .

docker-predict:
	docker run --rm -v $(pwd)/data:/data -v $(pwd)/output:/output \
		$(IMAGE) python predict.py --input /data/$(shell basename $(INPUT)) --output /output/$(shell basename $(OUTPUT))

clean:
	rm -rf output/ checkpoints/ runs/ __pycache__ src/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
