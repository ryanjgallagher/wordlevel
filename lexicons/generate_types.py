import argparse
import importlib
import pathlib
import warnings
from collections.abc import Iterable

from lexicons.huggingface import HuggingFaceUnavailableError
from lexicons.processor import LexiconProcessor


LEXICONS_DIR = pathlib.Path(__file__).parent.resolve()


def generate_changed_types(changed_fs: Iterable[str]) -> None:
    """
    Regenerates the types of every lexicon affected by the changed files. A lexicon whose raw data
    cannot be reached is skipped with a warning rather than held up, so that a commit can still be
    made without a network connection

    Args:
        changed_fs:
            The paths of the processor files that changed
    """
    processors = enumerate_processors()

    for lexicon in select_lexicons(changed_fs, all_lexicons=sorted(processors)):
        try:
            processors[lexicon]().generate_types()
        except HuggingFaceUnavailableError as error:
            warnings.warn(f"Skipped generating types for {lexicon}: {error}", stacklevel=2)


def enumerate_processors() -> dict[str, type[LexiconProcessor]]:
    """
    Finds all implemented processors (subclasses of LexiconProcessor)

    Returns:
        The processors, keyed by the name of the lexicon each one processes
    """
    for process_f in sorted(LEXICONS_DIR.glob("*/process.py")):
        importlib.import_module(f"lexicons.{process_f.parent.name}.process")

    return {processor().lexicon_name: processor for processor in LexiconProcessor.__subclasses__()}


def select_lexicons(changed_fs: Iterable[str], all_lexicons: list[str]) -> list[str]:
    """
    Determines which lexicons need their types regenerated, given the processor files that changed

    Args:
        changed_fs:
            The paths of the changed files, either `lexicons/processor.py` or
            `lexicons/{lexicon}/process.py`
        all_lexicons:
            The names of every lexicon that has a processor

    Returns:
        The names of the lexicons whose types need to be regenerated
    """
    changed_lexicons = set()

    for changed_f in changed_fs:
        changed_path = pathlib.Path(changed_f)

        # The base processor decides how every lexicon writes its types, so a change to it can
        # change any of them
        if changed_path.name == "processor.py":
            return all_lexicons

        if changed_path.parent.name in all_lexicons:
            changed_lexicons.add(changed_path.parent.name)

    return sorted(changed_lexicons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the parameter types of the lexicons whose processors changed"
    )

    parser.add_argument(
        "changed_fs",
        type=str,
        nargs="*",
        help="The paths of the processor files that changed",
    )

    generate_changed_types(parser.parse_args().changed_fs)
