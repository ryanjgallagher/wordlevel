"""Constants across the package"""

from wordlevel.configs import SHIFT_BAR_NAMES


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

SHIFT_SCORE_TYPES = list(vars(SHIFT_BAR_NAMES))
COUNTERACTED_SUFFIX = "_counteracted"

TOKEN = "token"
NORMALIZATION_SUFFIX = "__wl_normed"

IS_BORROWED_COL = "is_borrowed"

DEFAULT_SCORE_AXIS_PIXEL_SIZE = 400
