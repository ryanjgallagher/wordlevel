from process_lexicons import LexiconState, enumerate_lexicons, get_changed_lexicons

from lexicons.generate_types import enumerate_processors


def test_base_processor_change_reprocesses_every_lexicon() -> None:
    """
    Checks that a change to the base processor reprocesses every lexicon, since it decides how all
    of them are processed
    """
    lexicon_states = get_changed_lexicons(["lexicons/processor.py"])

    assert lexicon_states == [
        LexiconState(lexicon, processor_changed=True) for lexicon in enumerate_lexicons()
    ]


def test_base_processor_change_keeps_a_lexicons_own_changes() -> None:
    """
    Checks that a lexicon's own changes are not lost when the base processor changes alongside them
    """
    lexicon_states = get_changed_lexicons(["lexicons/processor.py", "lexicons/labMT/README.md"])

    assert LexiconState("labMT", processor_changed=True, readme_changed=True) in lexicon_states
    assert all(lexicon_state.processor_changed for lexicon_state in lexicon_states)


def test_enumerated_lexicons_match_the_implemented_processors() -> None:
    """
    Checks that the lexicon names given to the CI matrix are the names the processors use, since the
    matrix passes one name to both `lexicons.{name}.process` and `--lexicon {name}`
    """
    assert enumerate_lexicons() == sorted(enumerate_processors())
