# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Franklin D. Roosevelt", "Joe Biden"])

labmt = wl.lex.labMT("english")

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .alias("distinguishing"),
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(labmt, reference_score=labmt.center)
        .normalize(by="sum_abs")
        .alias("sentiment"),
)
# --8<-- [end:setup]


# --8<-- [start:bar_config]
chart = cl.plot.bar(
    "distinguishing",
    max_rank=15,
    show_totals=True,
    bar_config={
        "reference": "#0072B2",
        "comparison": "#E69F00",
        "reference_symbol": "◀",
        "comparison_symbol": "▶",
    },
)
# --8<-- [end:bar_config]


# --8<-- [start:score_sign_bar_config]
chart = cl.plot.bar(
    "sentiment",
    max_rank=15,
    show_totals=True,
    bar_config={"reinforce": "#0072B2", "offset": "#CC79A7"},
)
# --8<-- [end:score_sign_bar_config]


# --8<-- [start:shift_bar_config]
chart = cl.plot.shift(
    "sentiment",
    max_rank=15,
    bar_config={
        "more_of_positive": "#0072B2",
        "less_of_positive": "#56B4E9",
        "more_of_negative": "#D55E00",
        "less_of_negative": "#E69F00",
    },
)
# --8<-- [end:shift_bar_config]


# --8<-- [start:total_shift_config]
chart = cl.plot.shift(
    "sentiment",
    max_rank=15,
    show_totals=True,
    total_shift_config={"total": "#444444", "total_symbol": "Total"},
)
# --8<-- [end:total_shift_config]


# --8<-- [start:label_config]
chart = cl.plot.bar(
    "distinguishing",
    max_rank=15,
    label_config={"color": "#333333", "size": 13, "weight": "bold", "pad": 6},
)
# --8<-- [end:label_config]


# --8<-- [start:axis_config]
chart = cl.plot.bar(
    "distinguishing",
    max_rank=15,
    rank_axis={"title": "Position", "label_tick_step": 3},
    score_axis={"title": "Divergence", "grid": False},
)
# --8<-- [end:axis_config]


# --8<-- [start:border_config]
chart = cl.plot.bar(
    "distinguishing",
    max_rank=15,
    border_config={"color": "#333333", "width": 1},
    zero_line_config={"color": "#333333", "width": 2},
)
# --8<-- [end:border_config]


# Renders whichever chart was built most recently as an inline SVG
# --8<-- [start:render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:render]
