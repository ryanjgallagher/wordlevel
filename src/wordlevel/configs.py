"""Configuration classes for wordlevel"""

from dataclasses import dataclass, field, fields


@dataclass
class ShiftTypeStyle:
    """Visual style configuration for a single shift type"""

    color: str
    order: int
    name: str = field(default="", init=False)


@dataclass
class ShiftTypesConfig:
    """Configuration for all shift type visual styles"""

    more_of_negative: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#2F7CCE", order=1)
    )
    more_of_positive: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#FFFF80", order=1)
    )
    less_of_positive: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#FDFFD2", order=1)
    )
    less_of_negative: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#C4CAFC", order=1)
    )
    lower_score: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#9E75B7", order=2)
    )
    higher_score: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#FECC5D", order=2)
    )
    lower_score_counteracted: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#decee6", order=3)
    )
    higher_score_counteracted: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#ffedc7", order=3)
    )
    more_of_negative_counteracted: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#b6d1ee", order=3)
    )
    more_of_positive_counteracted: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#ffffd4", order=3)
    )
    less_of_positive_counteracted: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#fffff0", order=3)
    )
    less_of_negative_counteracted: ShiftTypeStyle = field(
        default_factory=lambda: ShiftTypeStyle(color="#ebedff", order=3)
    )

    def __post_init__(self):
        for f in fields(self):
            getattr(self, f.name).name = f.name

    def colors(self) -> dict[str, str]:
        """ """
        return {f.name: getattr(self, f.name).color for f in fields(self)}

    def orders(self) -> dict[str, int]:
        """ """
        return {f.name: getattr(self, f.name).order for f in fields(self)}
