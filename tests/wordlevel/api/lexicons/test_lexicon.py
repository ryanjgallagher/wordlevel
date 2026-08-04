import polars as pl
import pytest

from wordlevel.api.lexicons.lexicon import LexiconLoader
from wordlevel.core.lexicons.load import Lexicon


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

    monkeypatch.setattr("wordlevel.api.lexicons.lexicon.load_lexicon", fake_load_lexicon)

    return load_kwargs


@pytest.mark.parametrize("input_token", ["hf_abc123", None])
def test_hf_token_passed_to_load(load_kwargs: dict[str, object], input_token: str | None) -> None:
    """
    Checks that the token given to the loader reaches `load_lexicon`, since the loader is the only
    place a user can supply one. When no token is given, `None` is passed down, which is what lets
    `huggingface_hub` fall back to the `HF_TOKEN` environment variable or a local login.
    """
    LexiconLoader(hf_token=input_token).labMT("english")

    assert load_kwargs["hf_token"] == input_token
