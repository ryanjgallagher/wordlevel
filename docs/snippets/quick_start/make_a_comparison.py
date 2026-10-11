# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Franklin D. Roosevelt", "Joe Biden"])
# --8<-- [end:setup]


# Hide printing the shape
# --8<-- [start:display_config]
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_fmt_str_lengths(40)
# --8<-- [end:display_config]


# --8<-- [start:inspect_catalog]
print(cl.df.head(n=5))
# --8<-- [end:inspect_catalog]


# --8<-- [start:comparison]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .alias("roosevelt_biden"),
)
# --8<-- [end:comparison]


# --8<-- [start:top_tokens]
roosevelt_biden = cl.comparisons["roosevelt_biden"]

print(
    roosevelt_biden.df.select("token", "comparison_score", "score_rank", "attribution")
    .sort("score_rank")
    .head(n=10)
)
# --8<-- [end:top_tokens]


# --8<-- [start:bar_chart]
chart = cl.plot.bar(
    "roosevelt_biden",
    show_totals=True,
    totals_config={
        "reference_label": "Roosevelt",
        "comparison_label": "Biden",
    },
)
# --8<-- [end:bar_chart]


# Renders the chart built above as an inline SVG
# --8<-- [start:bar_chart_render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:bar_chart_render]
