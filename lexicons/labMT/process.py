import os

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
        languages = os.listdir(f"{self.base_dir}/data/raw")

        for language in languages:
            self._process_language(language=language)

    def _process_language(self, language: str) -> None:
        """ """
        words_f = f"{self.base_dir}/data/raw/{language}/labMTwords-{language}.csv"
        scores_f = f"{self.base_dir}/data/raw/{language}/labMTscores-{language}.csv"

        words = open(words_f).read().splitlines()
        scores = [float(score) for score in open(scores_f).read().splitlines()]

        df = pl.DataFrame({"token": words, "score": scores}).sort("score", descending=True)

        self._write_lexicon(
            df,
            filepath=f"{language}.parquet",
            metadata={"min": str(MIN_SCORE), "max": str(MAX_SCORE)},
        )

    def _generate_types(self) -> None:
        """ """
        languages = sorted(f.stem for f in (self.base_dir / "data" / "processed").glob("*.parquet"))

        self._write_literals(values=languages, variable="labMTLanguage")


if __name__ == "__main__":
    labMTProcessor().process()
