import pytest

from wordlevel.comparison.comparison import Comparison, comp
from wordlevel.comparison.exclude import ComparisonExcluder
from wordlevel.comparison.score import Scorer


def test_initialization_defaults() -> None:
    """Test that optional fields default to None or empty values."""
    comparison = Comparison("ref", "cmp")

    assert comparison._state.alias is None
    assert comparison._state.method is None
    assert comparison._state.is_weighted_avg is None
    assert comparison._state.attributed_by is None
    assert comparison._state.score_plan is None
    assert comparison._state.parameters == {}
    assert isinstance(comparison._state.parameters, dict)


def test_with_state_populates_internal_fields() -> None:
    """Test that `_with_state` stores every internal field."""

    def mock_function(df, token_col, normalization_suffix):
        return df

    mock_plan = mock_function
    params = {"alpha": 1.5, "base": 2}

    comparison = Comparison("ref_col", "cmp_col")._update_state(
        alias="my_alias",
        method="jensen_shannon",
        is_weighted_avg=True,
        attributed_by="frequency",
        score_plan=mock_plan,
        parameters=params,
    )

    assert comparison._state.reference == "ref_col"
    assert comparison._state.comparison == "cmp_col"
    assert comparison._state.alias == "my_alias"
    assert comparison._state.method == "jensen_shannon"
    assert comparison._state.is_weighted_avg is True
    assert comparison._state.attributed_by == "frequency"
    assert comparison._state.score_plan is mock_plan
    assert comparison._state.parameters == {"alpha": 1.5, "base": 2}
    assert comparison._state.parameters is not params  # Parameters are copied


@pytest.mark.parametrize("alias", ["alias1", "another_alias"])
def test_alias(alias: str) -> None:
    """Test that alias() returns a new Comparison with the specified alias."""
    comparison = Comparison("ref", "cmp")
    new_comparison = comparison.alias(alias)

    assert isinstance(new_comparison, Comparison)
    assert new_comparison is not comparison

    assert new_comparison._state.alias == alias
    assert comparison._state.alias is None

    assert new_comparison._state.reference == comparison._state.reference
    assert new_comparison._state.comparison == comparison._state.comparison


def test_alias_chaining() -> None:
    """Test that alias() can be called multiple times."""
    comparison = Comparison("ref", "cmp")

    new_comparison1 = comparison.alias("first")
    new_comparison2 = new_comparison1.alias("second")

    assert comparison._state.alias is None
    assert new_comparison1._state.alias == "first"
    assert new_comparison2._state.alias == "second"

    assert new_comparison2 is not new_comparison1
    assert new_comparison1 is not comparison


def test_alias_preserves_scoring_parameters() -> None:
    """Test that alias() preserves method, plan, and parameters from scoring."""
    comparison = Comparison("ref", "cmp")._update_state(
        method="jensen_shannon",
        parameters={"alpha": 1.0},
        is_weighted_avg=True,
        attributed_by="frequency",
    )

    new_comparison = comparison.alias("my_alias")

    assert new_comparison._state.method == "jensen_shannon"
    assert new_comparison._state.parameters == {"alpha": 1.0}
    assert new_comparison._state.is_weighted_avg is True
    assert new_comparison._state.attributed_by == "frequency"


def test_no_shared_state_of_parameters_between_instances() -> None:
    """Test that parameters dict isn't shared between instances."""
    comp1 = Comparison("ref1", "cmp1")
    comp2 = Comparison("ref2", "cmp2")

    comp1._state.parameters["key"] = "value"

    assert "key" not in comp2._state.parameters
    assert comp2._state.parameters == {}


def test_no_shared_state_of_parameters_when_chaining() -> None:
    """Test that parameters dict isn't shared between original and aliased instances."""
    comp1 = Comparison("ref", "cmp")._update_state(parameters={"alpha": 1.0})
    comp2 = comp1.alias("foo")

    comp2._state.parameters["alpha"] = 2.0

    # Check that the original comparison is unaffected
    assert comp1._state.parameters == {"alpha": 1.0}
    assert comp2._state.parameters == {"alpha": 2.0}


def test_external_mutation_of_parameters() -> None:
    """Test that external mutation of parameters dict doesn't affect Comparison."""
    params = {"alpha": 1.0}
    comparison = Comparison("ref", "cmp")._update_state(parameters=params)

    params["alpha"] = 2.0

    # Check that the original comparison is unaffected
    assert comparison._state.parameters == {"alpha": 1.0}


def test_scorer_integration() -> None:
    """ """
    comparison = Comparison("ref", "cmp")

    assert isinstance(comparison.score, Scorer)
    assert comparison.score._comparison is comparison


def test_excluder_integration() -> None:
    """ """
    comparison = Comparison("ref", "cmp")

    assert isinstance(comparison.exclude, ComparisonExcluder)
    assert comparison.exclude._comparison is comparison


@pytest.mark.parametrize("ref_col, cmp_col", [("frequency1", "frequency2"), ("freqs1", "freqs2")])
def test_constructor(ref_col: str, cmp_col: str) -> None:
    """ """
    comparison = comp(ref_col, cmp_col)

    assert isinstance(comparison, Comparison)
    assert comparison._state.reference == ref_col
    assert comparison._state.comparison == cmp_col
