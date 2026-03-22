from random import randint

import pytest
import polars as pl
from polars.testing import assert_frame_equal

from wordlevel.compare.comparator import CorpusComparator
from wordlevel.compare.types import ComparatorData
from wordlevel.compare.constructors import _normalize_frequency_columns
from wordlevel.compare.comparisons.comparison import ComparisonResult


@pytest.fixture
def label_col() -> str:
    return "word"


@pytest.fixture
def norm_suffix() -> str:
    return "_normed"


@pytest.fixture
def freq_cols() -> list[str]:
    return [f"freq{k}" for k in range(1, 5)]


@pytest.fixture
def words(freq_cols) -> list[str]:
    return "a b c d e f g h i j k l m n o p q r s t u v w x y z".split()[: len(freq_cols) + 1]


@pytest.fixture
def df_mock(label_col, words, freq_cols) -> pl.LazyFrame:
    """ """
    freqs = [[randint(1, 50) for _ in range(5)] for _ in freq_cols]

    return pl.DataFrame({label_col: words, **dict(zip(freq_cols, freqs))}).lazy()


@pytest.fixture
def mock_comparison_result(df_mock, freq_cols) -> ComparisonResult:
    return ComparisonResult(
        _lf=df_mock,
        reference=freq_cols[0],
        comparison=freq_cols[1],
        alias="test_comparison",
        method="test_method",
        is_weighted_avg=False,
        is_nonnegative=False,
        parameters={},
    )


@pytest.fixture
def mock_comparator_data(
    df_mock, mock_comparison_result, label_col, freq_cols, norm_suffix
) -> ComparatorData:
    return ComparatorData(
        df=df_mock,
        label_col=label_col,
        freq_cols=freq_cols,
        normalization_suffix=norm_suffix,
        comparisons=dict({"comparison": mock_comparison_result}),
    )


class TestConstructors:
    def test_init_from_comparator_data(
        self,
        df_mock,
        mock_comparison_result,
        mock_comparator_data,
        label_col,
        norm_suffix,
        freq_cols,
    ):
        """
        Checks a comparator is initialized correctly from comparator data. Does not check that the
        comparisons and dataframe are different objects because we assume ownership of copying is done
        by constructors
        """
        comparator = CorpusComparator(mock_comparator_data)

        assert comparator._label_col == label_col
        assert comparator._normalization_suffix == norm_suffix

        assert sorted(list(comparator._freq_cols)) == sorted(freq_cols)
        assert comparator._comparisons == {"comparison": mock_comparison_result}
        assert_frame_equal(comparator._df, df_mock, check_column_order=False, check_row_order=False)

    @pytest.mark.parametrize("constructor", [CorpusComparator, CorpusComparator.from_comparator])
    def test_from_comparator(self, mock_comparator_data, constructor):
        """Test initialization from an existing Comparator"""
        comp_original = CorpusComparator(data=mock_comparator_data)
        comp_new = constructor(comp_original)

        # Check that they are distinct class instances
        assert isinstance(comp_new, CorpusComparator)
        assert comp_original is not comp_new

        # Check the classes hold the same data
        assert comp_original._label_col == comp_new._label_col
        assert comp_original._normalization_suffix == comp_new._normalization_suffix
        assert sorted(comp_original._freq_cols) == sorted(comp_new._freq_cols)

        assert comp_original._comparisons == comp_new._comparisons
        assert comp_original._comparisons is not comp_new._comparisons

        assert isinstance(comp_new._df, pl.LazyFrame)
        assert_frame_equal(
            comp_original._df, comp_new._df, check_column_order=False, check_row_order=False
        )
        assert comp_original._df is not comp_new._df

    @pytest.mark.parametrize("constructor", [CorpusComparator, CorpusComparator.from_polars])
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("normalization_suffix", ["norm_suffix1", "norm_suffix2"])
    def test_from_polars(
        self, df_mock, label_col, freq_cols, normalize, normalization_suffix, constructor
    ):
        """Test initialization from a dataframe"""
        comparator = constructor(
            df_mock,
            label_col=label_col,
            freq_cols=freq_cols,
            normalize=normalize,
            normalization_suffix=normalization_suffix,
        )

        assert isinstance(comparator, CorpusComparator)

        assert comparator._label_col == label_col
        assert sorted(comparator._freq_cols) == sorted(freq_cols)
        assert isinstance(comparator._comparisons, dict)
        assert len(comparator._comparisons) == 0

        assert_frame_equal(
            comparator._df.select(label_col, *freq_cols).collect(),
            df_mock.collect(),
            check_row_order=False,
            check_column_order=False,
        )
        assert comparator._df is not df_mock

        if normalize:
            assert comparator._normalization_suffix == normalization_suffix
            for norm_freq_col in [f"{freq_col}{normalization_suffix}" for freq_col in freq_cols]:
                assert norm_freq_col in comparator._df
        else:
            assert comparator._normalization_suffix == ""


class TestConstructorValidation:
    def test_from_polars_duplicate_freq_cols(self, df_mock, freq_cols, label_col):
        comparator = CorpusComparator(
            data=df_mock, label_col=label_col, freq_cols=freq_cols + freq_cols
        )

        assert sorted(list(comparator._freq_cols)) == sorted(freq_cols)

    def test_from_polars_invalid_freq_cols(self, df_mock, freq_cols, label_col):
        """Test that from_polars throws an error if not all the columns are specified"""
        with pytest.raises(
            ValueError, match="`freq_cols` must be specified if providing a DataFrame"
        ):
            CorpusComparator(data=df_mock, label_col=label_col, freq_cols=None)

        with pytest.raises(KeyError, match="Frequency column not in provided dataframe"):
            CorpusComparator(data=df_mock, label_col=label_col, freq_cols=freq_cols + ["1000freq"])

        for not_enough_cols in [[], [freq_cols[0]]]:
            with pytest.raises(ValueError, match="At least two frequency columns"):
                CorpusComparator(data=df_mock, label_col=label_col, freq_cols=not_enough_cols)

    def test_from_polars_invalid_label_col(self, df_mock, freq_cols, label_col):
        """Test that from_polars throws an error if not all the columns are specified"""
        with pytest.raises(
            ValueError, match="`label_col` must be specified if providing a DataFrame"
        ):
            CorpusComparator(data=df_mock, label_col=None, freq_cols=freq_cols)

        with pytest.raises(KeyError, match="Label column not in provided dataframe"):
            CorpusComparator(data=df_mock, label_col="wordcol", freq_cols=freq_cols)


class TestConstructorMutation:
    def test_from_comparator_df_mutation(self, mock_comparator_data):
        """Test that mutating the original comparator's dataframe doesn't affect the copy."""
        comp_original = CorpusComparator(data=mock_comparator_data)
        comp_new = CorpusComparator(data=comp_original)

        # Get original row count
        n_rows_original = comp_new._df.collect().height

        # Mutate the original's dataframe by filtering
        comp_original._df = comp_original._df.filter(pl.col("word") != "a")

        # Verify the new comparator's dataframe is unchanged
        assert comp_new._df.collect().height == n_rows_original
        assert comp_original._df.collect().height != comp_new._df.collect().height

    def test_from_comparator_comparisons_mutation_independence(self, mock_comparator_data, df_mock):
        """Test that mutating the original comparator's comparisons dict doesn't affect the copy."""
        comp_original = CorpusComparator(data=mock_comparator_data)
        comp_new = CorpusComparator(data=comp_original)

        # Get original comparisons
        original_comparisons = comp_new._comparisons.copy()

        # Mutate the original's comparisons dict
        new_result = ComparisonResult(
            _lf=df_mock,
            reference="a",
            comparison="b",
            alias="new_comparison",
            method="test",
            is_weighted_avg=False,
            is_nonnegative=False,
            parameters={},
        )
        comp_original._comparisons["new_comparison"] = new_result

        # Verify the new comparator's comparisons dict is unchanged
        assert comp_new._comparisons == original_comparisons
        assert "new_comparison" not in comp_new._comparisons
        assert "new_comparison" in comp_original._comparisons


@pytest.mark.parametrize("normalization_suffix", ["suffix1", "suffix2"])
def test_normalize_frequency_columns(normalization_suffix):
    df_input = pl.DataFrame(
        {
            "label": ["a", "b", "c", "d"],
            "freqs1": [1, 1, 1, 1],
            "freqs2": [2, 3, 1, 4],
        }
    )
    df_expected = df_input.with_columns(
        pl.Series(f"freqs1{normalization_suffix}", [0.25, 0.25, 0.25, 0.25]),
        pl.Series(f"freqs2{normalization_suffix}", [0.2, 0.3, 0.1, 0.4]),
    )

    df_actual = _normalize_frequency_columns(
        df_input, freq_cols=["freqs1", "freqs2"], normalization_suffix=normalization_suffix
    )

    assert_frame_equal(df_actual, df_expected, check_column_order=False, check_row_order=False)
