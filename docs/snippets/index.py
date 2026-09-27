# The example shown on the home page. `docs/scripts/generate_cover_word_shift_graph.py` runs this
# file and saves `chart` as the cover image, so the code on the page and the image beneath it
# always come from the same source. The save lives in the generator, not here, so that the snippet
# test does not write images.
# --8<-- [start:example]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")

cl = wl.Catalog(speeches, corpora=["Franklin D. Roosevelt", "Joe Biden"])

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(
            lexicon_reference=wl.lex.SocialSent("1940"),
            lexicon_comparison=wl.lex.SocialSent("2000"),
            reference_score="center",
        )
        .normalize(by="total_diff")
        .alias("roosevelt_biden"),
)

chart = cl.plot.shift("roosevelt_biden", show_totals=True)
# --8<-- [end:example]