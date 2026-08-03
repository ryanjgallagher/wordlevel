"""
Imports every module in the installed `wordlevel` package

Meant to be run against an install built from the dependencies declared in `pyproject.toml`
alone, without the dev group or the lock file. A module that imports a package `wordlevel` never
declared still works in development, because something else pulls that package in transitively;
this fails on it instead of leaving it for a user to hit.
"""

import importlib
import pkgutil

import wordlevel


def main() -> int:
    modules = [module.name for module in pkgutil.walk_packages(wordlevel.__path__, "wordlevel.")]

    failures: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001
            failures.append(f"{module}: {type(error).__name__}: {error}")

    for failure in failures:
        print(f"FAILED {failure}")

    if failures:
        return 1

    print(f"Imported {len(modules)} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
