from lexicons.generate_types import select_lexicons


ALL_LEXICONS = ["labMT", "SocialSent"]


def test_select_lexicons_selects_the_changed_processors_lexicon() -> None:
    """
    Checks that a changed processor selects only the lexicon it processes
    """
    assert select_lexicons(["lexicons/labMT/process.py"], ALL_LEXICONS) == ["labMT"]


def test_select_lexicons_selects_every_lexicon_for_the_base_processor() -> None:
    """
    Checks that a change to the base processor selects every lexicon, since it decides how all of
    them write their types
    """
    assert select_lexicons(["lexicons/processor.py"], ALL_LEXICONS) == ALL_LEXICONS
