from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.datasets import dataset as dataset_module
from wordlevel.datasets.dataset import Dataset


DATA_DIR = Path(dataset_module.__file__).parent / "data"
DATASET_NAMES = sorted(p.stem for p in DATA_DIR.glob("*.parquet"))


@pytest.fixture(params=DATASET_NAMES)
def dataset(request) -> Dataset:
    return Dataset(name=request.param)


@pytest.fixture
def corpora(dataset) -> list[str]:
    """The first two corpora (non-token columns) of the dataset."""
    return [col for col in dataset.as_df().columns if col != "token"][:2]


def test_all_datasets_load():
    """Every dataset shipped in data/ loads via Dataset(name=...)."""
    # Guard against an empty data directory silently passing every parametrized test
    assert DATASET_NAMES

    for name in DATASET_NAMES:
        Dataset(name=name)


def test_as_df_returns_token_and_corpora(dataset):
    """as_df returns a token column plus at least two corpus columns."""
    df = dataset.as_df()

    assert "token" in df.columns
    assert len(df.columns) >= 3  # token + at least two corpora


def test_as_df_selects_corpora(dataset, corpora):
    """as_df(corpora=...) narrows to the token column plus the requested corpora."""
    df_all = dataset.as_df()
    df_subset = dataset.as_df(corpora=corpora)

    assert df_subset.columns == ["token", *corpora]

    # The selected columns are unchanged, just narrowed
    assert_frame_equal(df_subset, df_all.select("token", *corpora))


@pytest.mark.parametrize("select_subset", [False, True])
def test_as_dict_of_freqs_matches_as_df(dataset, select_subset, request):
    """as_dict_of_freqs round-trips back to as_df."""
    corpora = request.getfixturevalue("corpora") if select_subset else None

    df = dataset.as_df(corpora=corpora)
    corpus2token2freq = dataset.as_dict_of_freqs(corpora=corpora)

    tokens = df["token"].to_list()
    df_rebuilt = pl.DataFrame(
        {
            "token": tokens,
            **{
                corpus: [freqs[token] for token in tokens]
                for corpus, freqs in corpus2token2freq.items()
            },
        }
    )

    assert_frame_equal(df_rebuilt, df, check_dtypes=False)


@pytest.mark.parametrize("select_subset", [False, True])
def test_as_list_of_freqs_matches_as_df(dataset, select_subset, request):
    """as_list_of_freqs round-trips back to as_df."""
    corpora = request.getfixturevalue("corpora") if select_subset else None

    df = dataset.as_df(corpora=corpora)
    corpus_names, corpus_freqs = dataset.as_list_of_freqs(corpora=corpora)

    tokens = df["token"].to_list()
    df_rebuilt = pl.DataFrame(
        {
            "token": tokens,
            **{
                corpus: [freqs[token] for token in tokens]
                for corpus, freqs in zip(corpus_names, corpus_freqs)
            },
        }
    )

    assert_frame_equal(df_rebuilt, df, check_dtypes=False)
