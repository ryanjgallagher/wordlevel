import os
import pathlib

import polars as pl


BASE_DIR = pathlib.Path(__file__).parent.resolve()

MIN_SCORE = 1
CENTER_SCORE = 5
MAX_SCORE = 9


def process_lexicon(language: str) -> None:
    """
    Processes the raw labMT sentment data for a particular language into Parquet. Assumes that there
    are two files per language, one with the words, and one with the scores, and that the words and
    scores are ordered correspondingly.

    `data/raw/{language}/labMTwords-{language}.csv`
    `data/raw/{language}/labMTscores-{language}.csv`
    """

    words_f = f"{BASE_DIR}/data/raw/{language}/labMTwords-{language}.csv"
    scores_f = f"{BASE_DIR}/data/raw/{language}/labMTscores-{language}.csv"

    words = open(words_f).read().splitlines()
    scores = [float(score) for score in open(scores_f).read().splitlines()]

    df = pl.DataFrame({"token": words, "score": scores}).sort("score", descending=True)

    output_f = f"{BASE_DIR}/data/processed/{language}.parquet"

    df.write_parquet(
        output_f,
        metadata={"min": str(MIN_SCORE), "center": str(CENTER_SCORE), "max": str(MAX_SCORE)},
    )


if __name__ == "__main__":
    os.makedirs(f"{BASE_DIR}/data/processed", exist_ok=True)

    languages = os.listdir(f"{BASE_DIR}/data/raw")

    for language in languages:
        process_lexicon(language)
