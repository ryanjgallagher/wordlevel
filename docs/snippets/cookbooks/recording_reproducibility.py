# Mirrors the setup on the Score Comparisons page, plus a normalized comparison so the
# normalization metadata below is populated.
# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Joe Biden", "Franklin D. Roosevelt"])

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .normalize(by="sum_abs")
        .alias("roosevelt_biden_rank_divergence_normed"),
)
# --8<-- [end:setup]


# --8<-- [start:comparison_metadata]
comparison = cl.comparisons["roosevelt_biden_rank_divergence_normed"]

print(f"Reference corpus:  {comparison.reference}")
print(f"Comparison corpus: {comparison.comparison}")
print(f"Scoring method:    {comparison.method}")
print(f"Method parameters: {comparison.parameters}")
print(f"Attributed by:     {comparison.attributed_by}")
print(f"Exclusions:        {comparison.exclusions}")
# --8<-- [end:comparison_metadata]


# --8<-- [start:normalization_metadata]
print(f"Comparison is normalized: {comparison.normalized}")
print(f"Normalization method:     {comparison.normalized_by}")
print(f"Normalization constant:   {comparison.normalization_constant}")
# --8<-- [end:normalization_metadata]
