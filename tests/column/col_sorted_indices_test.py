from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_6
from tests.utils import interchange_to_pandas


@pytest.mark.xfail(strict=False)
def test_expression_sorted_indices_ascending(library: str) -> None:
    df = integer_dataframe_6(library)
    df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices()
    result = df.take(sorted_indices)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {
            "a": [2, 2, 1, 1, 1],
            "b": [1, 2, 3, 4, 4],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.xfail(strict=False)
def test_expression_sorted_indices_descending(library: str) -> None:
    df = integer_dataframe_6(library)
    df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices(ascending=False)
    result = df.take(sorted_indices)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 2, 1],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.xfail(strict=False)
def test_column_sorted_indices_ascending(library: str) -> None:
    df = integer_dataframe_6(library).persist()
    sorted_indices = pdx.col("b").sorted_indices()
    result = df.take(sorted_indices)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {
            "a": [2, 2, 1, 1, 1],
            "b": [1, 2, 3, 4, 4],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.xfail(strict=False)
def test_column_sorted_indices_descending(library: str) -> None:
    df = integer_dataframe_6(library).persist()
    sorted_indices = pdx.col("b").sorted_indices(ascending=False)
    result = df.take(sorted_indices)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 2, 1],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)
