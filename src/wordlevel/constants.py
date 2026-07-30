"""Constants across the package"""

from dataclasses import fields

from wordlevel.configs import ShiftBarConfig, SYMBOL_SUFFIX

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

SHIFT_SCORE_TYPES = [f.name for f in fields(ShiftBarConfig) if not f.name.endswith(SYMBOL_SUFFIX)]
COUNTERACTED_SUFFIX = "_counteracted"

TOKEN = "token"
NORMALIZATION_SUFFIX = "__wl_normed"

IS_BORROWED_COL = "is_borrowed"

DEFAULT_SCORE_AXIS_PIXEL_SIZE = 400
