.PHONY: docs-serve docs-build

zensical-serve:
	uv run python docs/scripts/gen_api_docs.py --output-config zensical-build.toml
	uv run zensical serve -f zensical-build.toml

zensical-build:
	uv run python docs/scripts/gen_api_docs.py --output-config zensical-build.toml
	uv run zensical build -f zensical-build.toml --clean
