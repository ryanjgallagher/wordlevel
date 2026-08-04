import argparse
import pathlib
from typing import Literal

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import (
    HTTPError,  # type: ignore[reportPrivateImportUsage]
    OfflineModeIsEnabled,
)


REPO = "ryanjgallag/wordlevel-lexicons"
BASE_DIR = pathlib.Path(__file__).parent.resolve()


class HuggingFaceUnavailableError(Exception):
    """
    Raised when the HuggingFace repository cannot be reached, as opposed to being reached and not
    holding the data that was asked for
    """


def get_hf_data_dir(lexicon: str, data_stage: Literal["raw", "processed"]) -> str:
    """
    The location of a lexicon's data within the HuggingFace repository
    """
    return f"{lexicon}/data/{data_stage}"


def list_lexicon_files(lexicon: str, data_stage: Literal["raw", "processed"]) -> list[str]:
    """
    Lists a lexicon's data files in the HuggingFace repository. Only the names of the files are
    read, so no data is downloaded

    Raises:
        HuggingFaceUnavailableError: If the repository cannot be reached
    """
    hf_data_dir = get_hf_data_dir(lexicon, data_stage)

    api = HfApi()
    try:
        repo_fs = api.list_repo_files(repo_id=REPO, repo_type="dataset")
    except (HTTPError, OfflineModeIsEnabled) as error:
        raise HuggingFaceUnavailableError(
            f"Could not reach the HuggingFace repository '{REPO}': {error}"
        ) from error

    return [repo_f for repo_f in repo_fs if repo_f.startswith(f"{hf_data_dir}/")]


def upload_lexicon(lexicon: str, data_stage: Literal["raw", "processed"]) -> None:
    """ """
    local_data_dir = BASE_DIR / lexicon / "data" / data_stage
    if not local_data_dir.exists() or not local_data_dir.is_dir():
        raise FileNotFoundError(f"The local directory '{local_data_dir}' does not exist")

    hf_data_dir = get_hf_data_dir(lexicon, data_stage)
    print(f"Uploading '{local_data_dir}' to {REPO} at '{hf_data_dir}'...")

    api = HfApi()
    api.upload_folder(
        folder_path=str(local_data_dir),
        path_in_repo=hf_data_dir,
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"Update of {lexicon} lexicon ({data_stage})",
    )

    print(f"Upload completed for {lexicon} ({data_stage})")


def upload_lexicon_metadata(lexicon: str, metadata: Literal["README.md", "LICENSE"]) -> None:
    """ """
    local_metadata_f = BASE_DIR / lexicon / metadata
    if not local_metadata_f.exists():
        raise FileNotFoundError(f"The local file '{local_metadata_f}' does not exist")

    hf_metadata_f = f"{lexicon}/{metadata}"
    print(f"Uploading '{local_metadata_f}' to {REPO} at '{hf_metadata_f}'...")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_metadata_f),
        path_in_repo=hf_metadata_f,
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"Update {metadata} for {lexicon} lexicon",
    )

    print(f"Upload completed for {lexicon} {metadata}")


def download_lexicon(lexicon: str, data_stage: Literal["raw", "processed"]) -> None:
    """ """
    hf_data_dir = get_hf_data_dir(lexicon, data_stage)

    print(f"Downloading '{hf_data_dir}' from {REPO}...")
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=f"{hf_data_dir}/**",
        local_dir=str(BASE_DIR),
    )

    local_data_dir = BASE_DIR / lexicon / "data" / data_stage
    print(f"Download completed for {lexicon} ({data_stage}) to '{local_data_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync lexicon data with HuggingFace via the command line"
    )

    parser.add_argument("--lexicon", type=str, required=True, help="The name of the lexicon")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["upload", "download"],
        required=True,
        help="Whether to 'upload' or 'download'",
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["raw", "processed"],
        required=False,
        dest="data_stage",
        help="The lexicon data to upload ('raw' or 'processed')",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        choices=["README.md", "LICENSE", "all"],
        required=False,
        help="The lexicon metadata to upload ('all', 'README.md', or 'LICENSE')",
    )

    args = parser.parse_args()

    if args.mode == "upload":
        if not args.data_stage and not args.metadata:
            parser.error("You must specify either --data or --metadata to perform an upload")

        if args.data_stage:
            upload_lexicon(lexicon=args.lexicon, data_stage=args.data_stage)

        if args.metadata:
            if args.metadata == "all":
                upload_lexicon_metadata(lexicon=args.lexicon, metadata="README.md")
                upload_lexicon_metadata(lexicon=args.lexicon, metadata="LICENSE")
            else:
                upload_lexicon_metadata(lexicon=args.lexicon, metadata=args.metadata)
    elif args.mode == "download":
        if args.metadata:
            parser.error("Downloading metadata is not currently supported")
        if not args.data_stage:
            parser.error("You must specify --data to perform a download")

        download_lexicon(lexicon=args.lexicon, data_stage=args.data_stage)
    else:
        parser.error("Invalid option given to --mode: please specify 'upload' or 'download'")
