from wordlevel.configs import BAR_OUTLINE_FIELDS
from wordlevel.constants import SHIFT_SCORE_TYPES


def test_shift_score_types_exclude_bar_outline_fields():
    """
    The bar outline fields on `ShiftBarConfig` style the bars and are not score types
    """
    assert BAR_OUTLINE_FIELDS.isdisjoint(SHIFT_SCORE_TYPES)
