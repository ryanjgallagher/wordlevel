from pathlib import Path
from typing import Collection

import polars as pl


class Dataset:
    def __init__(self, name: str) -> None:
        """
        Dataset loader to use for examples in the `wordlevel` package

        Args:
            name: Name of the dataset to load
        """
        data_f = Path(__file__).parent / "data" / f"{name}.parquet"
        self._df = pl.read_parquet(data_f)
        self._token_col = "token"

    def as_df(self, corpora: Collection[str] | None = None) -> pl.DataFrame:
        """
        Returns the dataset as a dataframe, where rows are tokens and columns corpora, and each
        entry indicates the frequency of that token in that corpus.

        Args:
            corpora: Names of the specific corpora to return from the dataset. If None, returns all
        """
        df = self._df
        if corpora is not None:
            df = df.select(self._token_col, *corpora)

        return df

    def as_dict_of_freqs(self, corpora: Collection[str] | None = None) -> dict[str, dict[str, int]]:
        """
        Returns the dataset as a dictionary of dictionaries {corpus: {token: frequency}}.

        Args:
            corpora: Names of the specific corpora to return from the dataset. If None, returns all
        """
        df = self.as_df(corpora=corpora)

        col2values = df.to_dict(as_series=False)
        tokens = col2values.pop(self._token_col)

        corpus2token2freq = {
            corpus: dict(zip(tokens, freqs)) for corpus, freqs in col2values.items()
        }

        return corpus2token2freq

    def as_list_of_freqs(
        self, corpora: Collection[str] | None = None
    ) -> tuple[list[str], list[dict[str, int]]]:
        """
        Returns the dataset as two lists: one of the corpus names and one of the {token: frequency}
        mappings.

        Args:
            corpora: Names of the specific corpora to return from the dataset. If None, returns all
        """
        df = self.as_df(corpora=corpora)

        col2values = df.to_dict(as_series=False)
        tokens = col2values.pop(self._token_col)

        corpus_names = []
        corpus_freqs = []
        for corpus, freqs in col2values.items():
            corpus_names.append(corpus)
            corpus_freqs.append(dict(zip(tokens, freqs)))

        return corpus_names, corpus_freqs
