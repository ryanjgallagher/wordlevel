import os

import polars as pl

from lexicons.processor import LexiconProcessor


class SocialSentProcessor(LexiconProcessor):
    """ """

    @property
    def lexicon_name(self) -> str:
        return "SocialSent"

    @property
    def center_score(self) -> float:
        return 0.0

    def _process_lexicon(self) -> None:
        """ """
        year_fs = os.listdir(f"{self.base_dir}/data/raw")
        years = sorted([year.split(".")[0] for year in year_fs if "README" not in year])

        for year in years:
            self._process_year(year=year)

    def _process_year(self, year: str) -> None:
        """ """
        scores_f = f"{self.base_dir}/data/raw/{year}.tsv"

        df = (
            pl.read_csv(scores_f, has_header=False, separator="\t")
            .select(pl.col("column_1").alias("token"), pl.col("column_2").alias("score"))
            .sort("score", descending=True)
        )

        min_score = df["score"].min()
        max_score = df["score"].max()

        self._write_lexicon(
            df, filepath=f"{year}.parquet", metadata={"min": str(min_score), "max": str(max_score)}
        )

    def _generate_types(self) -> None:
        years = sorted(f.stem for f in (self.base_dir / "data" / "processed").glob("*.parquet"))

        self._write_literals(values=years, variable="SocialSentYear")


if __name__ == "__main__":
    SocialSentProcessor().process()
