from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_6
from tests.utils import interchange_to_pandas


def test_expression_sorted_indices_ascending(library: str) -> None:
    df = integer_dataframe_6(library)
    df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices()
    result = df.get_rows(sorted_indices)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {
            "a": [2, 2, 1, 1, 1],
            "b": [1, 2, 3, 4, 4],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)


def test_expression_sorted_indices_descending(library: str) -> None:
    df = integer_dataframe_6(library)
    df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices(ascending=False)
    result = df.get_rows(sorted_indices)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 2, 1],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)


def test_column_sorted_indices_ascending(library: str) -> None:
    df = integer_dataframe_6(library).persist()
    sorted_indices = df.col("b").sorted_indices()
    result = df.get_rows(sorted_indices)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {
            "a": [2, 2, 1, 1, 1],
            "b": [1, 2, 3, 4, 4],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)


def test_column_sorted_indices_descending(library: str) -> None:
    df = integer_dataframe_6(library).persist()
    sorted_indices = df.col("b").sorted_indices(ascending=False)
    result = df.get_rows(sorted_indices)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 2, 1],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)
