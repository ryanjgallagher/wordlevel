# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Franklin D. Roosevelt", "Joe Biden"])

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.jsd()
        .normalize(by="total_diff")
        .alias("roosevelt_biden_jsd"),
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.shannon_entropy()
        .normalize(by="total_diff")
        .alias("roosevelt_biden_entropy"),
)
# --8<-- [end:setup]


# --8<-- [start:bar]
chart = cl.plot.bar("roosevelt_biden_jsd")
# --8<-- [end:bar]


# --8<-- [start:corpus_totals]
chart = cl.plot.bar("roosevelt_biden_jsd", max_rank=20, show_totals=True)
# --8<-- [end:corpus_totals]


# --8<-- [start:corpus_colors]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=20,
    show_totals=True,
    bar_config={
        "reference": "#009E73",
        "comparison": "#7B3294",
    },
    totals_config={
        "reference_label": "Roosevelt",
        "comparison_label": "Biden",
    },
)
# --8<-- [end:corpus_colors]


# --8<-- [start:sign_totals]
chart = cl.plot.bar("roosevelt_biden_entropy", max_rank=20, show_totals=True)
# --8<-- [end:sign_totals]


# --8<-- [start:sign_colors]
chart = cl.plot.bar(
    "roosevelt_biden_entropy",
    max_rank=20,
    show_totals=True,
    bar_config={
        "reinforce": "#D55E00",
        "offset": "#CC79A7",
    },
    totals_config={
        "reinforce_label": "Raises",
        "offset_label": "Lowers",
    },
)
# --8<-- [end:sign_colors]


# --8<-- [start:max_rank]
chart = cl.plot.bar("roosevelt_biden_jsd", max_rank=20)
# --8<-- [end:max_rank]


# --8<-- [start:width]
chart = cl.plot.bar("roosevelt_biden_jsd", max_rank=20, width=500)
# --8<-- [end:width]


# --8<-- [start:height]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=5,
    rank_axis_config={"bar_width": 24},
)
# --8<-- [end:height]


# --8<-- [start:horizontal_size]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=20,
    vertical=False,
    height=300,
    rank_axis_config={"bar_width": 24},
)
# --8<-- [end:horizontal_size]


# --8<-- [start:rotated_labels]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=20,
    vertical=False,
    label_config={"rotation": -45},
)
# --8<-- [end:rotated_labels]


# --8<-- [start:axes]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=20,
    rank_axis_config={
        "title": "Position",
        "title_color": "#7B3294",
        "label_tick_step": 3,
        "label_fontsize": 12,
    },
    score_axis_config={
        "title": "Percent contribution to the JSD",
        "title_fontsize": 14,
        "label_color": "#7B3294",
        "grid": True,
    },
)
# --8<-- [end:axes]


# --8<-- [start:labels]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=20,
    label_config={"color": "#333333", "size": 13, "weight": "bold", "pad": 6},
)
# --8<-- [end:labels]


# --8<-- [start:totals]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=20,
    show_totals=True,
    totals_config={"grand_total_color": "#D55E00", "grand_total_label": "JSD"},
)
# --8<-- [end:totals]


# --8<-- [start:borders]
chart = cl.plot.bar(
    "roosevelt_biden_jsd",
    max_rank=20,
    border_config={"color": "#7B3294", "width": 1},
    zero_line_config={"color": "#009E73", "width": 2},
)
# --8<-- [end:borders]


# Renders whichever chart was built most recently as an inline SVG
# --8<-- [start:render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:render]
