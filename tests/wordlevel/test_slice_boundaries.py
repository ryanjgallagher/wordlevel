import ast
import pathlib

import pytest


# Located by path rather than by importing the package, so that a structural problem is reported
# as a failure even when it also happens to break the import
SRC = pathlib.Path(__file__).parents[2] / "src" / "wordlevel"

CORE_DIRS = sorted(path for path in SRC.rglob("core") if path.is_dir())
SHARED_MODULES = sorted(path for path in SRC.glob("*.py") if path.name != "__init__.py")


def is_slice(name: str) -> bool:
    """Whether a top-level name is one of the package's slices, rather than a shared module"""
    return (SRC / name).is_dir()


def wordlevel_imports(path: pathlib.Path) -> set[str]:
    """
    The `wordlevel` modules a file imports, including those guarded by `TYPE_CHECKING`.

    Parsed rather than imported: every rule here is about code that imports perfectly well and is
    only structurally wrong, so there is nothing to catch at import time
    """
    imported: set[str] = set()

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    return {name for name in imported if name.startswith("wordlevel.")}


@pytest.mark.parametrize("core_dir", CORE_DIRS, ids=lambda path: path.parent.name)
def test_core_imports_only_core(core_dir: pathlib.Path) -> None:
    """
    A slice's `core` imports other slices' `core` and the shared modules, and nothing else.

    `core` holds the pure functions and the modules beside it hold the classes that compose them,
    so the dependency runs one way only. That applies across slices as much as within one: pure
    code should not reach for another slice's classes either, even in an annotation
    """
    for path in sorted(core_dir.rglob("*.py")):
        for module in wordlevel_imports(path):
            parts = module.removeprefix("wordlevel.").split(".")

            if is_slice(parts[0]) and parts[1:2] != ["core"]:
                pytest.fail(
                    f"{path.relative_to(SRC)} imports {module}, which is not core; core may "
                    f"import any slice's core and the shared modules, but not the classes built "
                    f"on it"
                )


@pytest.mark.parametrize("path", SHARED_MODULES, ids=lambda path: path.name)
def test_shared_modules_do_not_import_slices(path: pathlib.Path) -> None:
    """
    The shared top-level modules import nothing from the slices.

    Every slice is free to depend on them, so they have to stay a leaf. `__init__.py` is excluded,
    since re-exporting the public API is exactly what it is for
    """
    for module in wordlevel_imports(path):
        if is_slice(module.removeprefix("wordlevel.").split(".")[0]):
            pytest.fail(
                f"{path.name} imports {module}; the shared modules are what the slices are built "
                f"on, so they cannot depend on one"
            )
