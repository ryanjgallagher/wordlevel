from pathlib import Path

import polars as pl
import pytest

from wordlevel.core.lexicons.load import Lexicon


@pytest.fixture(autouse=True)
def mock_lexicons(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace load_lexicon with an in-memory stub so doc-snippet tests never call out to
    HuggingFace. The snippets only need a valid Lexicon to flow through scoring and
    plotting; the tokens below match the vocab used in the snippets so the lexicon join
    lands real scores.
    """

    def fake_load_lexicon(
        lexicon: str,
        filename: str,
        subfolder: str | None = None,
        cache_dir: str | Path | None = None,
        hf_token: str | None = None,
    ) -> Lexicon:
        return Lexicon(
            lf=pl.LazyFrame(
                {
                    "token": ["good", "bad", "happy", "sad"],
                    "score": [8.0, 2.0, 9.0, 1.0],
                }
            ),
            center=5.0,
        )

    monkeypatch.setattr("wordlevel.api.lexicons.lexicon.load_lexicon", fake_load_lexicon)
