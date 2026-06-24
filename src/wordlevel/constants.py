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

SHIFT_SCORE_TYPES = [f.name for f in fields(ShiftBarConfig) if not f.name.endswith(SYMBOL_SUFFIX)]
COUNTERACTED_SUFFIX = "_counteracted"

TOKEN = "token"
NORMALIZATION_SUFFIX = "__wl_normed"
