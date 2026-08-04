import os
import pathlib

import polars as pl

from lexicons.processor import LexiconProcessor


MIN_SCORE = 1.0
MAX_SCORE = 9.0


class labMTProcessor(LexiconProcessor):
    """
    Processes the raw labMT sentment data for a particular language into Parquet. Assumes that there
    are two files per language, one with the words, and one with the scores, and that the words and
    scores are ordered correspondingly.

    `data/raw/{language}/labMTwords-{language}.csv`
    `data/raw/{language}/labMTscores-{language}.csv`
    """

    @property
    def lexicon_name(self) -> str:
        return "labMT"

    @property
    def center_score(self) -> float:
        return 5.0

    def _process_lexicon(self) -> None:
        """ """
        languages = os.listdir(self.raw_dir)

        for language in languages:
            self._process_language(language=language)

    def _process_language(self, language: str) -> None:
        """ """
        words_f = self.raw_dir / language / f"labMTwords-{language}.csv"
        scores_f = self.raw_dir / language / f"labMTscores-{language}.csv"

        with open(words_f) as f:
            words = f.read().splitlines()

        with open(scores_f) as f:
            scores = [float(score) for score in f.read().splitlines()]

        df = pl.DataFrame({"token": words, "score": scores}).sort("score", descending=True)

        self._write_lexicon(
            df,
            filepath=f"{language}.parquet",
            metadata={"min": str(MIN_SCORE), "max": str(MAX_SCORE)},
        )

    def _generate_types(self, raw_fs: list[str]) -> None:
        """ """
        languages = {pathlib.PurePosixPath(raw_f).parent.name for raw_f in raw_fs}

        self._write_literals(values=languages, variable="labMTLanguage")


if __name__ == "__main__":
    labMTProcessor().process()
