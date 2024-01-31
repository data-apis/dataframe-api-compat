from __future__ import annotations

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_update_column(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    new_col = pdx.col("b") + 3
    result = df.assign(new_col)
    expected = {"a": [1, 2, 3], "b": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=pdx.Int64)


def test_update_columns(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    new_col_a = pdx.col("a") + 1
    new_col_b = pdx.col("b") + 3
    result = df.assign(new_col_a, new_col_b)
    expected = {"a": [2, 3, 4], "b": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=pdx.Int64)
