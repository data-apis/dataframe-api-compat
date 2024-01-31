from __future__ import annotations

import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_insert_columns(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    new_col = (pdx.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=pdx.Int64)


def test_insert_multiple_columns(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    new_col = (pdx.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=pdx.Int64)


def test_insert_multiple_columns_invalid(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    new_col = (pdx.col("b") + 3).rename("result")
    with pytest.raises(TypeError):
        _ = df.assign([new_col.rename("c"), new_col.rename("d")])  # type: ignore[arg-type]


def test_insert_eager_columns(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    new_col = (pdx.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=pdx.Int64)
