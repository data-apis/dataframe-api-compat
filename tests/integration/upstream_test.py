class TestPolars:
    def test_dataframe(self) -> None:
        import polars as pl

        df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df_pl.__dataframe_consortium_standard__()
        result = df.get_column_names()
        expected = ["a", "b"]
        assert result == expected

    def test_lazyframe(self) -> None:
        import polars as pl

        df_pl = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df_pl.__dataframe_consortium_standard__()
        result = df.get_column_names()
        expected_1 = ["a", "b"]
        assert result == expected_1

        ser = pl.Series([1, 2, 3])
        col = ser.__column_consortium_standard__()
        result = col.get_value(1)
        expected_2 = 2
        assert result == expected_2


class TestPandas:
    def test_pandas(self) -> None:
        """
        Test some basic methods of the dataframe consortium standard.

        Full testing is done at https://github.com/data-apis/dataframe-api-compat,
        this is just to check that the entry point works as expected.
        """
        import pandas as pd

        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df_pd.__dataframe_consortium_standard__()
        result_1 = df.get_column_names()
        expected_1 = ["a", "b"]
        assert result_1 == expected_1

        ser = pd.Series([1, 2, 3])
        col = ser.__column_consortium_standard__()
        result_2 = col.get_value(1)
        expected_2 = 2
        assert result_2 == expected_2
