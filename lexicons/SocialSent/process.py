import os
import pathlib

import polars as pl


BASE_DIR = pathlib.Path(__file__).parent.resolve()

CENTER_SCORE = 0


def process_lexicon(year: str) -> None:
    """
    Processes the raw SocialSent historical data for a particular decade into Parquet. Assumes that
    there is one TSV per year, `data/raw/{year}.tsv`
    """

    scores_f = f"{BASE_DIR}/data/raw/{year}.tsv"

    df = (
        pl.read_csv(scores_f, has_header=False, separator="\t")
        .select(pl.col("column_1").alias("token"), pl.col("column_2").alias("score"))
        .sort("score", descending=True)
    )

    output_f = f"{BASE_DIR}/data/processed/{year}.parquet"

    min_score = df["score"].min()
    max_score = df["score"].max()

    df.write_parquet(
        output_f,
        metadata={"min": str(min_score), "center": str(CENTER_SCORE), "max": str(max_score)},
    )


if __name__ == "__main__":
    os.makedirs(f"{BASE_DIR}/data/processed", exist_ok=True)

    year_fs = os.listdir(f"{BASE_DIR}/data/raw")
    years = sorted([year.split(".")[0] for year in year_fs if "README" not in year])

    for year in years:
        process_lexicon(year)
