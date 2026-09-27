# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Franklin D. Roosevelt", "Joe Biden"])

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(wl.lex.labMT("english"), reference_score="center")
        .normalize(by="total_diff")
        .alias("roosevelt_biden_labmt"),
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(
            lexicon_reference=wl.lex.SocialSent("1940"),
            lexicon_comparison=wl.lex.SocialSent("2000"),
            reference_score="center",
        )
        .normalize(by="total_diff")
        .alias("roosevelt_biden_socialsent"),
)
# --8<-- [end:setup]


# --8<-- [start:shift]
chart = cl.plot.shift("roosevelt_biden_labmt")
# --8<-- [end:shift]


# --8<-- [start:single_totals]
chart = cl.plot.shift("roosevelt_biden_labmt", max_rank=20, show_totals=True)
# --8<-- [end:single_totals]


# --8<-- [start:single_colors]
chart = cl.plot.shift(
    "roosevelt_biden_labmt",
    max_rank=20,
    show_totals=True,
    bar_config={
        "more_of_positive": "#009E73",
        "less_of_positive": "#A6DBC8",
        "more_of_negative": "#7B3294",
        "less_of_negative": "#C9A8D9",
    },
)
# --8<-- [end:single_colors]


# --8<-- [start:multiple_totals]
chart = cl.plot.shift("roosevelt_biden_socialsent", max_rank=20, show_totals=True)
# --8<-- [end:multiple_totals]


# --8<-- [start:multiple_colors]
chart = cl.plot.shift(
    "roosevelt_biden_socialsent",
    max_rank=20,
    show_totals=True,
    bar_config={
        "higher_score": "#D55E00",
        "lower_score": "#009E73",
        "higher_score_counteracted": "#F2BFA0",
        "lower_score_counteracted": "#A6DBC8",
    },
)
# --8<-- [end:multiple_colors]


# Renders whichever chart was built most recently as an inline SVG
# --8<-- [start:render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:render]
