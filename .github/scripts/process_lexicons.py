import os
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


LEXICONS_DIR = Path(__file__).parents[2] / "lexicons"


@dataclass
class LexiconState:
    """
    How a lexicon changed, which determines whether it is reprocessed and what is uploaded for it
    """

    name: str
    processor_changed: bool = False
    readme_changed: bool = False
    license_changed: bool = False


def get_changed_lexicons(changed_fs: list[str]) -> list[LexiconState]:
    """
    Determines which lexicons changed and how

    Args:
        changed_fs:
            The paths of the files changed in the commit

    Returns:
        The state of every lexicon that needs to be processed or uploaded
    """
    lexicon_states: dict[str, LexiconState] = {}

    # The base processor decides how every lexicon is processed, so a change to it means all of
    # them need reprocessing, not just the ones whose own files changed
    if any(Path(changed_f).name == "processor.py" for changed_f in changed_fs):
        lexicon_states = {
            lexicon: LexiconState(lexicon, processor_changed=True)
            for lexicon in enumerate_lexicons()
        }

    for changed_f in changed_fs:
        # lexicons/{lexicon}/{changed_f}
        path_parts = Path(changed_f).parts

        # The length check ensures it's a file in `lexicons/*/{lexicon}` and not just `lexicons/*`
        if not (path_parts[0] == "lexicons" and len(path_parts) >= 3):
            continue

        lexicon = path_parts[1]
        file_name = path_parts[2]

        if lexicon not in lexicon_states:
            lexicon_states[lexicon] = LexiconState(lexicon)

        if file_name == "README.md":
            lexicon_states[lexicon].readme_changed = True
        elif file_name == "LICENSE":
            lexicon_states[lexicon].license_changed = True
        elif file_name == "process.py":
            lexicon_states[lexicon].processor_changed = True

    return list(lexicon_states.values())


def enumerate_lexicons() -> list[str]:
    """
    Finds the names of all the lexicons that have a processor. The names come from the directories
    rather than from the processors themselves so that this can run without the package installed
    """
    return sorted(process_f.parent.name for process_f in LEXICONS_DIR.glob("*/process.py"))


def set_github_output(key: str, value: str) -> None:
    """
    Writes a key-value pair to the GitHub Actions output file. If no such file exists in the
    environment, just prints to the local standard out
    """
    github_output_path = os.environ.get("GITHUB_OUTPUT")

    if github_output_path:
        with open(github_output_path, "a") as outfile:
            outfile.write(f"{key}={value}\n")
    else:
        print(f"{key}={value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse changed files to determine which lexicons need processing"
    )
    parser.add_argument(
        "changed_files",
        type=str,
        help="A space-separated string of file paths changed in the commit",
    )

    args = parser.parse_args()
    lexicon_states = get_changed_lexicons(args.changed_files.split())

    # Output directly to the GitHub Actions $GITHUB_OUTPUT environment
    lexicon_changes = [asdict(lexicon_state) for lexicon_state in lexicon_states]
    set_github_output("lexicon_changes", json.dumps(lexicon_changes))
