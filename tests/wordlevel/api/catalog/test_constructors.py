from random import randint, sample

import pytest
import polars as pl
from polars.testing import assert_frame_equal

from wordlevel.api.catalog.catalog import Catalog
from wordlevel.constants import TOKEN, NORMALIZATION_SUFFIX
from wordlevel.api.catalog.constructors import CatalogData
from wordlevel.api.catalog.constructors import _normalize_frequency_columns
from wordlevel.api.comparison.comparison import ComparisonResult


@pytest.fixture
def token_col() -> str:
    return "word"


@pytest.fixture
def norm_suffix() -> str:
    return "_normed"


@pytest.fixture
def freq_cols() -> list[str]:
    return [f"freq{k}" for k in range(1, 5)]


@pytest.fixture
def words(freq_cols) -> list[str]:
    return "a b c d e f g h i j k l m n o p q r s t u v w x y z".split()


@pytest.fixture
def df_mock(token_col, words, freq_cols) -> pl.LazyFrame:
    """ """
    freqs = [[randint(1, 50) for _ in range(len(words))] for _ in freq_cols]

    return pl.DataFrame({token_col: words, **dict(zip(freq_cols, freqs))}).lazy()


@pytest.fixture
def df_mock_with_nulls(df_mock, freq_cols) -> pl.LazyFrame:
    """ """
    return df_mock.with_columns(
        pl.when(pl.int_range(1, pl.len() + 1) % (col_n + 2) == 0)
        .then(None)
        .otherwise(pl.col(freq_col))
        .alias(freq_col)
        for col_n, freq_col in enumerate(freq_cols)
    )


@pytest.fixture
def get_mock_freqs(words):
    def mock_freqs() -> dict[str, int]:
        """ """
        n_words = randint(1, len(words))
        word_indxs = sample(range(0, len(words)), n_words)

        corpus_words = [words[indx] for indx in word_indxs]
        corpus_freqs = [randint(1, 50) for _ in range(n_words)]

        return dict(zip(corpus_words, corpus_freqs))

    return mock_freqs


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
def mock_catalog_data(
    df_mock, mock_comparison_result, token_col, freq_cols, norm_suffix
) -> CatalogData:
    return CatalogData(
        df=df_mock,
        token_col=token_col,
        freq_cols=freq_cols,
        normalization_suffix=norm_suffix,
        comparisons=dict({"comparison": mock_comparison_result}),
    )


class TestConstructors:
    def test_init_from_catalog_data(
        self,
        df_mock,
        mock_comparison_result,
        mock_catalog_data,
        token_col,
        norm_suffix,
        freq_cols,
    ):
        """
        Checks a catalog is initialized correctly from catalog data. Does not check that the
        comparisons and dataframe are different objects because we assume ownership of copying is done
        by constructors
        """
        catalog = Catalog(mock_catalog_data)

        assert catalog._token_col == token_col
        assert catalog._normalization_suffix == norm_suffix

        assert sorted(list(catalog._freq_cols)) == sorted(freq_cols)
        assert catalog._comparisons == {"comparison": mock_comparison_result}
        assert_frame_equal(catalog._df, df_mock, check_column_order=False, check_row_order=False)

    @pytest.mark.parametrize("constructor", [Catalog, Catalog.from_catalog])
    def test_from_catalog(self, mock_catalog_data, constructor):
        """Test initialization from an existing Catalog"""
        cl_original = Catalog(data=mock_catalog_data)
        cl_new = constructor(cl_original)

        # Check that they are distinct class instances
        assert isinstance(cl_new, Catalog)
        assert cl_original is not cl_new

        # Check the classes hold the same data
        assert cl_original._token_col == cl_new._token_col
        assert cl_original._normalization_suffix == cl_new._normalization_suffix
        assert sorted(cl_original._freq_cols) == sorted(cl_new._freq_cols)

        assert cl_original._comparisons == cl_new._comparisons
        assert cl_original._comparisons is not cl_new._comparisons

        assert isinstance(cl_new._df, pl.LazyFrame)
        assert_frame_equal(
            cl_original._df, cl_new._df, check_column_order=False, check_row_order=False
        )
        assert cl_original._df is not cl_new._df

    @pytest.mark.parametrize("normalize", [False, True])
    def test_from_polars(self, df_mock, token_col, freq_cols, normalize):
        """Test initialization from a dataframe via the from_polars classmethod."""
        catalog = Catalog.from_polars(
            df_mock,
            token_col=token_col,
            freq_cols=freq_cols,
            normalize=normalize,
        )

        assert isinstance(catalog, Catalog)

        assert catalog._token_col == token_col
        assert sorted(catalog._freq_cols) == sorted(freq_cols)
        assert isinstance(catalog._comparisons, dict)
        assert len(catalog._comparisons) == 0

        assert_frame_equal(
            catalog._df.select(token_col, *freq_cols).collect(),
            df_mock.collect(),
            check_row_order=False,
            check_column_order=False,
        )
        assert catalog._df is not df_mock

        if normalize:
            assert catalog._normalization_suffix == NORMALIZATION_SUFFIX
            for norm_freq_col in [f"{freq_col}{NORMALIZATION_SUFFIX}" for freq_col in freq_cols]:
                assert norm_freq_col in catalog._df.collect_schema()
        else:
            assert catalog._normalization_suffix == ""

    @pytest.mark.parametrize("constructor", [Catalog, Catalog.from_polars])
    def test_from_polars_always_fills_nulls(
        self, df_mock_with_nulls, token_col, freq_cols, constructor
    ):
        """Frequencies with nulls are always filled to 0; the API offers no opt-out."""
        catalog = constructor(
            df_mock_with_nulls,
            token_col=token_col,
            freq_cols=freq_cols,
        )

        null_counts = catalog._df.select(pl.col(*freq_cols).is_null().any()).collect().row(0)
        assert not any(null_counts)

    @pytest.mark.parametrize("constructor", [Catalog, Catalog.from_list_of_freqs])
    @pytest.mark.parametrize("corpora", [None, ["ref", "comp1", "comp2"]])
    def test_from_list_of_freqs(self, get_mock_freqs, corpora, constructor):
        """Test initialization from a list of frequency dicts"""
        n_corpora = 3
        freqs = [get_mock_freqs() for _ in range(n_corpora)]
        expected_corpora = (
            corpora if corpora is not None else [f"corpus{n}" for n in range(1, n_corpora + 1)]
        )

        catalog = constructor(freqs, corpora=corpora)

        assert isinstance(catalog, Catalog)
        assert isinstance(catalog._comparisons, dict)
        assert len(catalog._comparisons) == 0

        # Check token column
        assert catalog._token_col == TOKEN

        # Check freq cols are named correctly using the custom prefix
        assert catalog._freq_cols == set(expected_corpora)

        # Check all words from all corpora are present (outer join)
        all_words = set().union(*[f.keys() for f in freqs])
        assert set(catalog.df[TOKEN].to_list()) == all_words

        # Normalization is always applied for the freq constructors
        assert catalog._normalization_suffix == NORMALIZATION_SUFFIX
        normed_freq_cols = [f"{freq_col}{NORMALIZATION_SUFFIX}" for freq_col in expected_corpora]
        for normed_freq_col in normed_freq_cols:
            assert normed_freq_col in catalog._df.collect_schema()

        # Check no nulls in any frequency or normed frequency columns
        cols_to_check = list(expected_corpora) + normed_freq_cols
        null_counts = catalog._df.select(pl.col(*cols_to_check).is_null().any()).collect().row(0)
        assert not any(null_counts)

    @pytest.mark.parametrize("constructor", [Catalog, Catalog.from_dict_of_freqs])
    def test_from_dict_of_freqs(self, get_mock_freqs, constructor):
        """Test initialization from a dict of frequency dicts"""
        corpora_names = ["ref", "comp1", "comp2"]
        freqs = {name: get_mock_freqs() for name in corpora_names}

        catalog = constructor(freqs)

        assert isinstance(catalog, Catalog)
        assert isinstance(catalog._comparisons, dict)
        assert len(catalog._comparisons) == 0

        # Check token column
        assert catalog._token_col == TOKEN

        # Check freq cols are named correctly using the custom prefix
        assert catalog._freq_cols == set(corpora_names)

        # Check all words from all corpora are present (outer join)
        all_words = set().union(*[f.keys() for f in freqs.values()])
        assert set(catalog.df[TOKEN].to_list()) == all_words

        # Normalization is always applied for the freq constructors
        assert catalog._normalization_suffix == NORMALIZATION_SUFFIX
        normed_freq_cols = [f"{freq_col}{NORMALIZATION_SUFFIX}" for freq_col in corpora_names]
        for normed_freq_col in normed_freq_cols:
            assert normed_freq_col in catalog._df.collect_schema()

        # Check no nulls in any frequency or normed frequency columns
        cols_to_check = list(corpora_names) + normed_freq_cols
        null_counts = catalog._df.select(pl.col(*cols_to_check).is_null().any()).collect().row(0)
        assert not any(null_counts)


class TestConstructorValidation:
    def test_from_polars_duplicate_freq_cols(self, df_mock, freq_cols, token_col):
        catalog = Catalog(data=df_mock, token_col=token_col, freq_cols=freq_cols + freq_cols)

        assert sorted(list(catalog._freq_cols)) == sorted(freq_cols)

    def test_from_polars_invalid_freq_cols(self, df_mock, freq_cols, token_col):
        """Test that from_polars throws an error if not all the columns are specified"""
        with pytest.raises(
            ValueError, match="`freq_cols` must be specified if providing a DataFrame"
        ):
            Catalog(data=df_mock, token_col=token_col, freq_cols=None)

        with pytest.raises(KeyError, match="Frequency column not in provided dataframe"):
            Catalog(data=df_mock, token_col=token_col, freq_cols=freq_cols + ["1000freq"])

        for not_enough_cols in [[], [freq_cols[0]]]:
            with pytest.raises(ValueError, match="At least two frequency columns"):
                Catalog(data=df_mock, token_col=token_col, freq_cols=not_enough_cols)

    def test_from_polars_invalid_token_col(self, df_mock, freq_cols, token_col):
        """Test that from_polars throws an error if not all the columns are specified"""
        with pytest.raises(
            ValueError, match="`token_col` must be specified if providing a DataFrame"
        ):
            Catalog(data=df_mock, token_col=None, freq_cols=freq_cols)

        with pytest.raises(KeyError, match="Token column not in provided dataframe"):
            Catalog(data=df_mock, token_col="wordcol", freq_cols=freq_cols)

    def test_from_list_of_freqs_invalid(self, get_mock_freqs):
        """Test that from_list_of_freqs raises errors for invalid inputs"""
        mock = get_mock_freqs()

        with pytest.raises(ValueError, match="At least two frequency sets"):
            Catalog([mock])

        with pytest.raises(ValueError, match="`freqs` and `corpora` must be the same length"):
            Catalog([mock, mock], corpora=["only_one"])

        with pytest.raises(ValueError, match="There are no token frequencies"):
            Catalog([mock, {}])

    def test_from_dict_of_freqs_invalid(self, get_mock_freqs):
        """Test that from_dict_of_freqs raises errors for invalid inputs"""
        mock = get_mock_freqs()

        with pytest.raises(ValueError, match="At least two frequency sets"):
            Catalog({"corpus1": mock})

        with pytest.raises(ValueError, match="There are no token frequencies"):
            Catalog({"corpus1": mock, "corpus2": {}})


class TestConstructorMutation:
    def test_from_catalog_df_mutation(self, mock_catalog_data):
        """Test that mutating the original catalog's dataframe doesn't affect the copy."""
        cl_original = Catalog(data=mock_catalog_data)
        cl_new = Catalog(data=cl_original)

        # Get original row count
        n_rows_original = cl_new._df.collect().height

        # Mutate the original's dataframe by filtering
        cl_original._df = cl_original._df.filter(pl.col("word") != "a")

        # Verify the new catalog's dataframe is unchanged
        assert cl_new._df.collect().height == n_rows_original
        assert cl_original._df.collect().height != cl_new._df.collect().height

    def test_from_catalog_comparisons_mutation_independence(self, mock_catalog_data, df_mock):
        """Test that mutating the original catalog's comparisons dict doesn't affect the copy."""
        cl_original = Catalog(data=mock_catalog_data)
        cl_new = Catalog(data=cl_original)

        # Get original comparisons
        original_comparisons = cl_new._comparisons.copy()

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
        cl_original._comparisons["new_comparison"] = new_result

        # Verify the new catalog's comparisons dict is unchanged
        assert cl_new._comparisons == original_comparisons
        assert "new_comparison" not in cl_new._comparisons
        assert "new_comparison" in cl_original._comparisons


@pytest.mark.parametrize("normalization_suffix", ["suffix1", "suffix2"])
def test_normalize_frequency_columns(normalization_suffix):
    df_input = pl.DataFrame(
        {
            "token": ["a", "b", "c", "d"],
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
