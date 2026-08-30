# --8<-- [start:setup]
import wordlevel as wl

before = {"good": 40, "bad": 20, "happy": 30, "sad": 10, "the": 100}
after = {"good": 20, "bad": 45, "happy": 15, "sad": 35, "the": 100}

cl = wl.Catalog({"before": before, "after": after})
# --8<-- [end:setup]


# Hide printing the shape
# --8<-- [start:display_config]
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)
# --8<-- [end:display_config]


# --8<-- [start:load_labmt]
labmt = wl.lex.labMT("english")
# --8<-- [end:load_labmt]


# --8<-- [start:inspect_lexicon]
print(f"Neutral score: {labmt.center}")
print(labmt.df.head(n=5))
# --8<-- [end:inspect_lexicon]


# --8<-- [start:score_with_lexicon]
cl = cl.with_comparisons(
    wl.comp("before", "after")
        .score.lexicon(labmt, reference_score=labmt.center)
        .alias("sentiment"),
)
# --8<-- [end:score_with_lexicon]


# --8<-- [start:custom_lexicon]
# Our own 1-5 scale, where 3 is neutral
custom_lexicon = {"good": 4.5, "bad": 1.5, "happy": 5.0, "sad": 1.0}

cl = cl.with_comparisons(
    wl.comp("before", "after")
        .score.lexicon(custom_lexicon, reference_score=3.0)
        .alias("custom_sentiment"),
)
# --8<-- [end:custom_lexicon]


# --8<-- [start:per_corpus_lexicons]
# One lexicon per corpus: each corpus is scored by its own
scores_before = {"good": 8.0, "bad": 2.0, "happy": 9.0}
scores_after = {"good": 7.5, "bad": 1.5, "sad": 1.0}

cl = cl.with_comparisons(
    wl.comp("before", "after")
        .score.lexicon(
            lexicon_reference=scores_before,
            lexicon_comparison=scores_after,
            reference_score=5.0,
        )
        .alias("drifting_sentiment"),
)
# --8<-- [end:per_corpus_lexicons]


# --8<-- [start:scored_tokens]
drifting = cl.comparisons["drifting_sentiment"]

print(drifting.df.select("token", "comparison_score").sort("token"))
# --8<-- [end:scored_tokens]
