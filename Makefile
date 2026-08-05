.PHONY: zensical-serve zensical-build docs-build strip-sources

zensical-serve:
	uv run python docs/scripts/gen_api_docs.py --output-config zensical-build.toml
	uv run zensical serve -f zensical-build.toml

zensical-build:
	uv run python docs/scripts/gen_api_docs.py --output-config zensical-build.toml
	uv run zensical build -f zensical-build.toml --clean
	$(MAKE) strip-sources

# Site without the API reference, built straight from zensical.toml. Used by CI.
# TODO: retire this target once API generation runs in CI; point the workflow at
# zensical-build instead.
docs-build:
	uv run zensical build --clean
	$(MAKE) strip-sources

# Zensical copies docs_dir verbatim, which would otherwise publish
# docs/scripts/*.py, docs/snippets/**/*.py and docs/overrides/main.html as live
# URLs. The find is rot-proof against new .py files landing under docs/.
strip-sources:
	find site -name "*.py" -delete
	find site -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf site/overrides
