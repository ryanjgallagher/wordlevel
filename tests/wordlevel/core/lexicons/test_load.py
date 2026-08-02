from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.core.lexicons.load import Lexicon, load_lexicon


CENTER = 5.0


@pytest.fixture
def lexicon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lexicon:
    """
    Writes a lexicon parquet the way `LexiconProcessor.write` does (scores in the frame, center
    score in the parquet's key-value metadata) and stubs out the HuggingFace download so
    `load_lexicon` reads that file instead of calling out to the hub
    """
    lexicon_f = tmp_path / "lexicon.parquet"
    pl.DataFrame({"token": ["good", "bad"], "score": [8.0, 2.0]}).write_parquet(
        lexicon_f, metadata={"center": str(CENTER)}
    )

    monkeypatch.setattr(
        "wordlevel.core.lexicons.load.hf_hub_download", lambda **kwargs: str(lexicon_f)
    )

    return load_lexicon("labMT", "lexicon.parquet")


def test_center_read_from_parquet_metadata(lexicon: Lexicon) -> None:
    """
    Checks that the center score is read out of the parquet's key-value metadata and cast to a
    float, rather than being left as the string it is stored as
    """
    assert lexicon.center == CENTER


def test_scores_read_from_parquet(lexicon: Lexicon) -> None:
    """
    Checks that the lexicon's scores are scanned from the same parquet the center comes from
    """
    expected = pl.DataFrame({"token": ["good", "bad"], "score": [8.0, 2.0]})

    assert_frame_equal(lexicon.df, expected)
