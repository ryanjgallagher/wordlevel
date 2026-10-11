"""Constants across the package"""

from types import SimpleNamespace


RANK_COL = "score_rank"
SCORE_COL = "comparison_score"
ATTRIBUTION_COL = "attribution"
NORMALIZATION_CONSTANT_COL = "normalization_constant"

PLOT_CONTRIBUTION_COL = "contribution"
PLOT_CONTRIBUTION_TYPE_COL = "contribution_type"

REFERENCE_ATTRIBUTION = "reference"
COMPARISON_ATTRIBUTION = "comparison"
REINFORCE_ATTRIBUTION = "reinforce"
OFFSET_ATTRIBUTION = "offset"

FREQ_REF_COL = "__wl_freq_ref"
FREQ_CMP_COL = "__wl_freq_cmp"
NORMED_FREQ_REF_COL = "__wl_normed_freq_ref"
NORMED_FREQ_CMP_COL = "__wl_normed_freq_cmp"
LEXICON_SCORE_REF_COL = "__wl_lexicon_score_ref"
LEXICON_SCORE_CMP_COL = "__wl_lexicon_score_cmp"

GRAND_TOTAL_SCORE_TYPE = "total"

COUNTERACTED_SUFFIX = "_counteracted"
SHIFT_BASE_SCORE_TYPES = (
    "more_of_negative",
    "more_of_positive",
    "less_of_positive",
    "less_of_negative",
    "lower_score",
    "higher_score",
)
SHIFT_SCORE_TYPES = [
    *SHIFT_BASE_SCORE_TYPES,
    *(f"{name}{COUNTERACTED_SUFFIX}" for name in SHIFT_BASE_SCORE_TYPES),
]
SHIFT_SCORE_TYPE_NAMES = SimpleNamespace(**{name: name for name in SHIFT_BASE_SCORE_TYPES})

TOKEN = "token"
NORMALIZATION_SUFFIX = "__wl_normed"

IS_BORROWED_COL = "is_borrowed"

DEFAULT_SCORE_AXIS_PIXEL_SIZE = 400
