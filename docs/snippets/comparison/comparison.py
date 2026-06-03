# --8<-- [start:comp]
import wordlevel as wl

wl.comp(reference="before", comparison="after")
# --8<-- [end:comp]


# --8<-- [start:alias]
import wordlevel as wl

wl.comp("before", "after").alias("my_comparison")
# --8<-- [end:alias]
