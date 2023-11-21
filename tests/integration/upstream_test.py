import sys

import pytest


class TestPolars:
    def test_dataframe(self) -> None:
        import polars as pl

        if tuple(int(v) for v in pl.__version__.split(".")) < (
            0,
            19,
            0,
        ):  # pragma: no cover
            # before consortium standard in polars
            return

        df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df_pl.__dataframe_consortium_standard__()
        result = df.get_column_names()
        expected = ["a", "b"]
        assert result == expected

    def test_lazyframe(self) -> None:
        import polars as pl

        if tuple(int(v) for v in pl.__version__.split(".")) < (
            0,
            19,
            0,
        ):  # pragma: no cover
            # before consortium standard in polars
            return

        df_pl = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df_pl.__dataframe_consortium_standard__()
        result = df.get_column_names()
        expected_1 = ["a", "b"]
        assert result == expected_1

        ser = pl.Series("a", [1, 2, 3])
        col = ser.__column_consortium_standard__()
        assert col.name == "a"


@pytest.mark.skipif(
    sys.version_info < (3, 9) or sys.version_info >= (3, 12),
    reason="pandas doesn't support 3.8",
)
class TestPandas:
    def test_pandas(self) -> None:
        """
        Test some basic methods of the dataframe consortium standard.

        Full testing is done at https://github.com/data-apis/dataframe-api-compat,
        this is just to check that the entry point works as expected.
        """
        import pandas as pd

        if tuple(int(v) for v in pd.__version__.split(".")) < (
            2,
            1,
            0,
        ):  # pragma: no cover
            # before consortium standard in pandas
            return

        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df_pd.__dataframe_consortium_standard__()
        result_1 = df.get_column_names()
        expected_1 = ["a", "b"]
        assert result_1 == expected_1

        ser = pd.Series([1, 2, 3], name="a")
        assert ser.name == "a"
