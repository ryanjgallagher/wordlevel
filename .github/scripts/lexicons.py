import os
import argparse
import json
from pathlib import Path


def main():
    """ """
    parser = argparse.ArgumentParser(
        description="Parse changed files to determine which lexicons need processing"
    )
    parser.add_argument(
        "changed_files",
        type=str,
        help="A space-separated string of file paths changed in the commit",
    )

    get_changed_lexicons(parser.parse_args())


def get_changed_lexicons(args: argparse.Namespace) -> None:
    """ """
    if not args.changed_files.strip():
        set_github_output("matrix_array", "[]")
        return

    lexicon_states = {}
    changed_fs = args.changed_files.split()
    for f_path in changed_fs:
        # lexicons/{lexicon}/{changed_f}
        path_parts = Path(f_path).parts

        # The length check ensures it's a file in `lexicons/*/{lexicon}` and not just `lexicons/*`
        if not (path_parts[0] == "lexicons" and len(path_parts) >= 3):
            continue

        lexicon = path_parts[1]
        file_name = path_parts[2]

        if lexicon not in lexicon_states:
            lexicon_states[lexicon] = {
                "name": lexicon,
                "processor_changed": False,
                "readme_changed": False,
                "license_changed": False,
            }

        if file_name == "README.md":
            lexicon_states[lexicon]["readme_changed"] = True
        elif file_name == "LICENSE":
            lexicon_states[lexicon]["license_changed"] = True
        elif file_name == "process.py":
            lexicon_states[lexicon]["processor_changed"] = True

    # Output directly to the GitHub Actions $GITHUB_OUTPUT environment
    set_github_output("lexicon_changes", json.dumps(list(lexicon_states.values())))


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
    main()
