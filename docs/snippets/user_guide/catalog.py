# --8<-- [start:setup]
import polars as pl
import wordlevel as wl

animals = {"the": 92, "and": 74, "duck": 20, "dog": 14, "cat": 11, "lion": 4}
cats = {"the": 90, "and": 78, "cat": 55, "lion": 38, "tabby": 20, "tuxedo": 19}
# --8<-- [end:setup]


# --8<-- [start:from_multiple_dicts]
cl = wl.Catalog([animals, cats], corpora=["animals", "cats"])
# --8<-- [end:from_multiple_dicts]


# --8<-- [start:from_single_dict]
cl = wl.Catalog({"animals": animals, "cats": cats})
# --8<-- [end:from_single_dict]


# --8<-- [start:build_dataframe]
# To create a dataframe, we need ordered columns, filling in 0s if a token doesn't appear in a corpus
tokens = sorted(list(set(animals) | set(cats)))
animal_freqs = [animals[token] if token in animals else 0 for token in tokens]
cat_freqs = [cats[token] if token in cats else 0 for token in tokens]

df_corpora = pl.DataFrame({"token": tokens, "animals": animal_freqs, "cats": cat_freqs})
# --8<-- [end:build_dataframe]


# --8<-- [start:from_dataframe]
cl = wl.Catalog(df_corpora, token_col="token", corpora=["animals", "cats"])
# --8<-- [end:from_dataframe]


# Hide printing the shape
# --8<-- [start:display_config]
pl.Config.set_tbl_hide_dataframe_shape(True)
# --8<-- [end:display_config]

# --8<-- [start:inspect]
print(cl.df.head(n=10).sort("token"))
# --8<-- [end:inspect]
