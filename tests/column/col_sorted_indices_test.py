from __future__ import annotations

import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_6


@pytest.mark.xfail(strict=False)
def test_expression_sorted_indices_ascending(library: str) -> None:
    df = integer_dataframe_6(library)
    pdx = df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices()
    result = df.take(sorted_indices)
    expected = {"a": [2, 2, 1, 1, 1], "b": [1, 2, 3, 4, 4]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


@pytest.mark.xfail(strict=False)
def test_expression_sorted_indices_descending(library: str) -> None:
    df = integer_dataframe_6(library)
    pdx = df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices(ascending=False)
    result = df.take(sorted_indices)
    expected = {"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 2, 1]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


@pytest.mark.xfail(strict=False)
def test_column_sorted_indices_ascending(library: str) -> None:
    df = integer_dataframe_6(library).persist()
    pdx = df.__dataframe_namespace__()
    sorted_indices = pdx.col("b").sorted_indices()
    result = df.take(sorted_indices)
    expected = {"a": [2, 2, 1, 1, 1], "b": [1, 2, 3, 4, 4]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


@pytest.mark.xfail(strict=False)
def test_column_sorted_indices_descending(library: str) -> None:
    df = integer_dataframe_6(library).persist()
    pdx = df.__dataframe_namespace__()
    sorted_indices = pdx.col("b").sorted_indices(ascending=False)
    result = df.take(sorted_indices)
    expected = {"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 2, 1]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
