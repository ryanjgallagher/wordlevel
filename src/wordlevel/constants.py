"""Constants across the package"""

from dataclasses import fields

from wordlevel.configs import ShiftTypesConfig

RANK_COL = "rank"
SCORE_COL = "comparison_score"
ATTRIBUTION_COL = "attribution"

PLOT_CONTRIBUTION_COL = "contribution"
PLOT_CONTRIBUTION_TYPE_COL = "contribution_type"

REFERENCE_ATTRIBUTION = "reference"
COMPARISON_ATTRIBUTION = "comparison"

SHIFT_SCORE_TYPES = [f.name for f in fields(ShiftTypesConfig)]
COUNTERACTED_SUFFIX = "_counteracted"
