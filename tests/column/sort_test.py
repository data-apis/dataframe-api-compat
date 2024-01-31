from __future__ import annotations

import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_6


@pytest.mark.xfail(strict=False)
def test_expression_sort_ascending(library: str) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    s_sorted = pdx.col("b").sort().rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [1, 2, 3, 4, 4],
    }
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


@pytest.mark.xfail(strict=False)
def test_expression_sort_descending(library: str) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    pdx = df.__dataframe_namespace__()
    s_sorted = pdx.col("b").sort(ascending=False).rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [4, 4, 3, 2, 1],
    }
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


@pytest.mark.xfail(strict=False)
def test_column_sort_ascending(library: str) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta").persist()
    pdx = df.__dataframe_namespace__()
    s_sorted = pdx.col("b").sort().rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [1, 2, 3, 4, 4],
    }
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


@pytest.mark.xfail(strict=False)
def test_column_sort_descending(library: str) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta").persist()
    pdx = df.__dataframe_namespace__()
    s_sorted = pdx.col("b").sort(ascending=False).rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [4, 4, 3, 2, 1],
    }
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
