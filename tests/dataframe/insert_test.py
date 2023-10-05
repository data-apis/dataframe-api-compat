from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_insert_columns(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    namespace = df.__dataframe_namespace__()
    new_col = (namespace.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"))
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    pd.testing.assert_frame_equal(result_pd, expected)
    # check original df didn't change
    df_pd = interchange_to_pandas(df, library)
    df_pd = convert_dataframe_to_pandas_numpy(df_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(df_pd, expected)


def test_insert_multiple_columns(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    namespace = df.__dataframe_namespace__()
    new_col = (namespace.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    )
    pd.testing.assert_frame_equal(result_pd, expected)
    # check original df didn't change
    df_pd = interchange_to_pandas(df, library)
    df_pd = convert_dataframe_to_pandas_numpy(df_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(df_pd, expected)


def test_insert_eager_columns(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    new_col = (df.collect().get_column_by_name("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    )
    pd.testing.assert_frame_equal(result_pd, expected)
    # check original df didn't change
    df_pd = interchange_to_pandas(df, library)
    df_pd = convert_dataframe_to_pandas_numpy(df_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(df_pd, expected)


def test_insert_reduction(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    new_col = (namespace.col("b").mean()).rename("result")
    result = df.assign(new_col)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "result": [5.0, 5.0, 5.0]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_insert_reduction_and_column(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.assign(col("b").mean().rename("c"), col("b").rename("d"))
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [5.0, 5.0, 5.0], "d": [4, 5, 6]}
    )
    pd.testing.assert_frame_equal(result_pd, expected)
