# Bar Graphs

A bar graph shows the tokens that contribute most to a comparison, ordered by the magnitude of their
contribution.

```python exec="true" source="above" session="bar_graphs"
--8<-- "snippets/how_to/bar_graphs.py:setup"
```

## Making a bar graph

Pass the alias of a comparison attached to the `Catalog`. The `max_rank` sets how many tokens are
shown.

```python exec="true" source="above" session="bar_graphs"
--8<-- "snippets/how_to/bar_graphs.py:bar"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="bar_graphs"
--8<-- "snippets/how_to/bar_graphs.py:bar_render"
```
