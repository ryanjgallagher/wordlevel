# Mirrors the setup on the "Make a Comparison" page: markdown-exec sessions do not
# cross page boundaries, so every page rebuilds its own catalog.
# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Franklin D. Roosevelt", "Joe Biden"])
# --8<-- [end:setup]


# --8<-- [start:load_lexicon]
labmt = wl.lex.labMT("english")
# --8<-- [end:load_lexicon]


# --8<-- [start:comparison]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(labmt, reference_score=labmt.center)
        .alias("roosevelt_biden_sentiment"),
)
# --8<-- [end:comparison]


# --8<-- [start:shift_chart]
chart = cl.plot.shift("roosevelt_biden_sentiment", show_totals=True)
# --8<-- [end:shift_chart]


# Renders the chart built above as an inline SVG
# --8<-- [start:shift_chart_render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:shift_chart_render]
