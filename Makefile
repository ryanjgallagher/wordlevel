.PHONY: zensical-serve zensical-build strip-sources

# `.cache` is Zensical's build cache. Its incremental rebuilds don't track snippet
# includes as page dependencies, so editing docs/snippets/**/*.py leaves the pages
# that include them stale, even across a restart. `serve` has no `--clean` flag, so
# clear the cache here to force a cold render.
zensical-serve:
	uv run python -m docs.scripts.api_docs --output-config zensical-build.toml
	rm -rf .cache
	uv run zensical serve -f zensical-build.toml

zensical-build:
	uv run python -m docs.scripts.api_docs --output-config zensical-build.toml
	uv run zensical build -f zensical-build.toml --clean --strict
	$(MAKE) strip-sources

# Zensical copies docs_dir verbatim, which would otherwise publish
# docs/scripts/*.py, docs/snippets/**/*.py and docs/overrides/main.html as live
# URLs. The find is rot-proof against new .py files landing under docs/.
strip-sources:
	find site -name "*.py" -delete
	find site -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf site/overrides
	find site -type d -empty -delete
