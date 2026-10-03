from pathlib import Path

import polars as pl
import pytest

from wordlevel.lexicons.core.load import Lexicon, LexiconConnectionError
from wordlevel.lexicons.lexicon import LexiconLoader


STORED_COMMIT = "a" * 40
LATEST_COMMIT = "b" * 40


@pytest.fixture
def load_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """
    Stubs out `load_lexicon` so that the arguments the loader passes down to it can be inspected
    without calling out to the hub
    """
    load_kwargs: dict[str, object] = {}

    def fake_load_lexicon(**kwargs: object) -> Lexicon:
        load_kwargs.update(kwargs)
        return Lexicon(lf=pl.LazyFrame({"token": ["good"], "score": [8.0]}), center=5.0)

    monkeypatch.setattr("wordlevel.lexicons.lexicon.load_lexicon", fake_load_lexicon)

    return load_kwargs


def stub_stored_revision(monkeypatch: pytest.MonkeyPatch, commit: str | None) -> None:
    """
    Stubs out `stored_revision` so that the lexicon appears to be stored at `commit`, or not stored
    at all if `commit` is `None`
    """
    monkeypatch.setattr("wordlevel.lexicons.lexicon.stored_revision", lambda **kwargs: commit)


def stub_resolve_revision(monkeypatch: pytest.MonkeyPatch, commit: str) -> list[str]:
    """
    Stubs out `resolve_revision` so that every revision resolves to `commit`, returning the list
    that the revisions asked of the hub are recorded in
    """
    requested_revisions: list[str] = []

    def fake_resolve_revision(**kwargs: object) -> str:
        requested_revisions.append(str(kwargs["revision"]))
        return commit

    monkeypatch.setattr("wordlevel.lexicons.lexicon.resolve_revision", fake_resolve_revision)

    return requested_revisions


def stub_unreachable_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Stubs out `resolve_revision` so that HuggingFace cannot be reached
    """

    def fake_resolve_revision(**kwargs: object) -> str:
        raise LexiconConnectionError("Could not reach HuggingFace")

    monkeypatch.setattr("wordlevel.lexicons.lexicon.resolve_revision", fake_resolve_revision)


@pytest.mark.parametrize("input_token", ["hf_abc123", None])
def test_hf_token_passed_to_load(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch, input_token: str | None
) -> None:
    """
    Checks that the token given to the loader reaches `load_lexicon`, since the loader is the only
    place a user can supply one. When no token is given, `None` is passed down, which is what lets
    `huggingface_hub` fall back to the `HF_TOKEN` environment variable or a local login.
    """
    stub_stored_revision(monkeypatch, STORED_COMMIT)

    LexiconLoader(hf_token=input_token).labMT("english")

    assert load_kwargs["hf_token"] == input_token


def test_cached_loads_stored_commit_without_checking_hub(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that by default a stored lexicon is loaded at the commit it was stored at, without
    asking HuggingFace whether there is a newer version
    """
    stub_stored_revision(monkeypatch, STORED_COMMIT)
    requested_revisions = stub_resolve_revision(monkeypatch, LATEST_COMMIT)

    LexiconLoader(revision="cached").labMT("english")

    assert requested_revisions == []
    assert load_kwargs["revision"] == STORED_COMMIT


def test_cached_downloads_latest_when_not_stored(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that by default a lexicon that has not been stored yet is downloaded at the latest
    commit of the repository's default branch
    """
    stub_stored_revision(monkeypatch, None)
    requested_revisions = stub_resolve_revision(monkeypatch, LATEST_COMMIT)

    LexiconLoader(revision="cached").labMT("english")

    assert requested_revisions == ["main"]
    assert load_kwargs["revision"] == LATEST_COMMIT


def test_latest_checks_hub_when_stored(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that asking for the latest version loads the latest commit of the repository's default
    branch, even when an older copy of the lexicon is stored
    """
    stub_stored_revision(monkeypatch, STORED_COMMIT)
    requested_revisions = stub_resolve_revision(monkeypatch, LATEST_COMMIT)

    LexiconLoader(revision="latest").labMT("english")

    assert requested_revisions == ["main"]
    assert load_kwargs["revision"] == LATEST_COMMIT


def test_latest_warns_and_loads_stored_commit_when_hub_unreachable(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that when the latest version is asked for but HuggingFace cannot be reached, the stored
    copy is loaded, and the user is warned that it may not be the latest version
    """
    stub_stored_revision(monkeypatch, STORED_COMMIT)
    stub_unreachable_hub(monkeypatch)

    with pytest.warns(UserWarning, match="Could not reach HuggingFace"):
        LexiconLoader(revision="latest").labMT("english")

    assert load_kwargs["revision"] == STORED_COMMIT


def test_latest_raises_when_hub_unreachable_and_not_stored(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that when the latest version is asked for, HuggingFace cannot be reached, and there is
    no stored copy to fall back to, the connection error is raised
    """
    stub_stored_revision(monkeypatch, None)
    stub_unreachable_hub(monkeypatch)

    with pytest.raises(LexiconConnectionError):
        LexiconLoader(revision="latest").labMT("english")


def test_pinned_commit_loads_stored_copy_without_checking_hub(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that a pinned commit that the lexicon is already stored at is loaded without asking
    HuggingFace, so that pinned lexicons load offline
    """
    stub_stored_revision(monkeypatch, STORED_COMMIT)
    requested_revisions = stub_resolve_revision(monkeypatch, LATEST_COMMIT)

    LexiconLoader(revision=STORED_COMMIT).labMT("english")

    assert requested_revisions == []
    assert load_kwargs["revision"] == STORED_COMMIT


@pytest.mark.parametrize("pinned_revision", [LATEST_COMMIT, "v1.0"])
def test_pinned_revision_resolved_when_stored_at_other_commit(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch, pinned_revision: str
) -> None:
    """
    Checks that a pinned commit, branch, or tag that the lexicon is not stored at is asked of
    HuggingFace, and the lexicon loaded at the commit it points to
    """
    stub_stored_revision(monkeypatch, STORED_COMMIT)
    requested_revisions = stub_resolve_revision(monkeypatch, LATEST_COMMIT)

    LexiconLoader(revision=pinned_revision).labMT("english")

    assert requested_revisions == [pinned_revision]
    assert load_kwargs["revision"] == LATEST_COMMIT


def test_pinned_revision_raises_when_hub_unreachable(
    load_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that a pinned revision never falls back to a stored copy at a different commit, since
    that would silently load a different version of the lexicon than the one asked for
    """
    stub_stored_revision(monkeypatch, STORED_COMMIT)
    stub_unreachable_hub(monkeypatch)

    with pytest.raises(LexiconConnectionError):
        LexiconLoader(revision=LATEST_COMMIT).labMT("english")


def test_update_cached_updates_stored_lexicons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that updating hands the stored lexicons, along with the loader's cache directory and
    token, down to be updated
    """
    repo_paths = ["labMT/data/processed/english.parquet"]
    update_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        "wordlevel.lexicons.lexicon.stored_lexicon_paths", lambda **kwargs: repo_paths
    )
    monkeypatch.setattr(
        "wordlevel.lexicons.lexicon.update_lexicons",
        lambda **kwargs: update_kwargs.update(kwargs),
    )

    LexiconLoader(cache_dir=tmp_path, hf_token="hf_abc123").update_cached()

    assert update_kwargs == {
        "cache_dir": tmp_path,
        "repo_paths": repo_paths,
        "hf_token": "hf_abc123",
    }


def test_update_cached_skips_hub_when_nothing_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that updating with no stored lexicons does not call out to HuggingFace, so that it
    succeeds offline when there is nothing to update
    """
    update_kwargs: dict[str, object] = {}

    monkeypatch.setattr("wordlevel.lexicons.lexicon.stored_lexicon_paths", lambda **kwargs: [])
    monkeypatch.setattr(
        "wordlevel.lexicons.lexicon.update_lexicons",
        lambda **kwargs: update_kwargs.update(kwargs),
    )

    LexiconLoader(cache_dir=tmp_path).update_cached()

    assert update_kwargs == {}
