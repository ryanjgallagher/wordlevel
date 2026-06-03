import pytest

from wordlevel.api.comparison.comparison import Comparison, comp
from wordlevel.api.comparison.score import Scorer


def test_initialization_defaults() -> None:
    """Test that optional fields default to None or empty values."""
    comparison = Comparison("ref", "cmp")

    assert comparison._alias is None
    assert comparison._method is None
    assert comparison._is_weighted_avg is None
    assert comparison._plan is None
    assert comparison._parameters == {}
    assert isinstance(comparison._parameters, dict)


def test_with_state_populates_internal_fields() -> None:
    """Test that `_with_state` stores every internal field."""

    def mock_function(df, token_col, normalization_suffix):
        return df

    mock_plan = mock_function
    params = {"alpha": 1.5, "base": 2}

    comparison = Comparison._with_state(
        reference="ref_col",
        comparison="cmp_col",
        alias="my_alias",
        method="jensen_shannon",
        is_weighted_avg=True,
        plan=mock_plan,
        parameters=params,
    )

    assert comparison._ref == "ref_col"
    assert comparison._cmp == "cmp_col"
    assert comparison._alias == "my_alias"
    assert comparison._method == "jensen_shannon"
    assert comparison._is_weighted_avg is True
    assert comparison._plan is mock_plan
    assert comparison._parameters == {"alpha": 1.5, "base": 2}
    assert comparison._parameters is not params  # Parameters are copied


@pytest.mark.parametrize("alias", ["alias1", "another_alias"])
def test_alias(alias: str) -> None:
    """Test that alias() returns a new Comparison with the specified alias."""
    comparison = Comparison("ref", "cmp")
    new_comparison = comparison.alias(alias)

    assert isinstance(new_comparison, Comparison)
    assert new_comparison is not comparison

    assert new_comparison._alias == alias
    assert comparison._alias is None

    assert new_comparison._ref == comparison._ref
    assert new_comparison._cmp == comparison._cmp


def test_alias_chaining() -> None:
    """Test that alias() can be called multiple times."""
    comparison = Comparison("ref", "cmp")

    new_comparison1 = comparison.alias("first")
    new_comparison2 = new_comparison1.alias("second")

    assert comparison._alias is None
    assert new_comparison1._alias == "first"
    assert new_comparison2._alias == "second"

    assert new_comparison2 is not new_comparison1
    assert new_comparison1 is not comparison


def test_alias_preserves_scoring_parameters() -> None:
    """Test that alias() preserves method, plan, and parameters from scoring."""
    comparison = Comparison._with_state(
        "ref", "cmp", method="jensen_shannon", parameters={"alpha": 1.0}, is_weighted_avg=True
    )

    new_comparison = comparison.alias("my_alias")

    assert new_comparison._method == "jensen_shannon"
    assert new_comparison._parameters == {"alpha": 1.0}
    assert new_comparison._is_weighted_avg is True


def test_no_shared_state_of_parameters_between_instances() -> None:
    """Test that parameters dict isn't shared between instances."""
    comp1 = Comparison("ref1", "cmp1")
    comp2 = Comparison("ref2", "cmp2")

    comp1._parameters["key"] = "value"

    assert "key" not in comp2._parameters
    assert comp2._parameters == {}


def test_no_shared_state_of_parameters_when_chaining() -> None:
    """Test that parameters dict isn't shared between original and aliased instances."""
    comp1 = Comparison._with_state("ref", "cmp", parameters={"alpha": 1.0})
    comp2 = comp1.alias("foo")

    comp2._parameters["alpha"] = 2.0

    # Check that the original comparison is unaffected
    assert comp1._parameters == {"alpha": 1.0}
    assert comp2._parameters == {"alpha": 2.0}


def test_external_mutation_of_parameters() -> None:
    """Test that external mutation of parameters dict doesn't affect Comparison."""
    params = {"alpha": 1.0}
    comparison = Comparison._with_state("ref", "cmp", parameters=params)

    params["alpha"] = 2.0

    # Check that the original comparison is unaffected
    assert comparison._parameters == {"alpha": 1.0}


def test_scorer_integration() -> None:
    """ """
    comparison = Comparison("ref", "cmp")

    assert isinstance(comparison.score, Scorer)
    assert comparison.score._comparison is comparison


@pytest.mark.parametrize("ref_col, cmp_col", [("frequency1", "frequency2"), ("freqs1", "freqs2")])
def test_constructor(ref_col: str, cmp_col: str) -> None:
    """ """
    comparison = comp(ref_col, cmp_col)

    assert isinstance(comparison, Comparison)
    assert comparison._ref == ref_col
    assert comparison._cmp == cmp_col
