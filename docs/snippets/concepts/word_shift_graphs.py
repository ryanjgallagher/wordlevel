# --8<-- [start:shift]
import wordlevel as wl

cl = (
    wl.Catalog(
        wl.Dataset("presidential_speeches"),
        corpora=["Franklin D. Roosevelt", "Joe Biden"],
    )
    .with_comparisons(
        wl.comp("Franklin D. Roosevelt", "Joe Biden")
            .score.lexicon(wl.lex.labMT("english"), reference_score="center")
            .alias("roosevelt_biden_sentiment"),
    )
)

chart = cl.plot.shift("roosevelt_biden_sentiment", show_totals=True)
# --8<-- [end:shift]


# --8<-- [start:detailed_shift]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(
            lexicon_reference=wl.lex.SocialSent("1940"),
            lexicon_comparison=wl.lex.SocialSent("2000"),
            reference_score="center",
        )
        .alias("roosevelt_biden_sentiment_by_decade"),
)

chart = cl.plot.shift("roosevelt_biden_sentiment_by_decade", show_totals=True)
# --8<-- [end:detailed_shift]


# Renders whichever chart was built most recently as an inline SVG
# --8<-- [start:render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:render]
