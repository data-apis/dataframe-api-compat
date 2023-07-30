from dataframe_api_compat.polars_standard import convert_to_standard_compliant_dataframe
import pandas as pd
import polars as pl


def test_column_column() -> None:
    result = (
        convert_to_standard_compliant_dataframe(pl.DataFrame({"a": [1, 2, 3]}))
        .get_column_by_name("a")
        .column
    )
    pd.testing.assert_series_equal(result.to_pandas(), pd.Series([1, 2, 3], name="a"))
    result = (
        convert_to_standard_compliant_dataframe(pd.DataFrame({"a": [1, 2, 3]}))
        .get_column_by_name("a")
        .column
    )
    pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], name="a"))
