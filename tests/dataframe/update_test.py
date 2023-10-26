from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_update_column(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    df.__dataframe_namespace__()
    new_col = df.col("b") + 3
    result = df.assign(new_col)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [7, 8, 9]})
    pd.testing.assert_frame_equal(result_pd, expected)
    # check original df didn't change
    df_pd = interchange_to_pandas(df)
    df_pd = convert_dataframe_to_pandas_numpy(df_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(df_pd, expected)


def test_update_columns(library: str) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    df.__dataframe_namespace__()
    new_col_a = df.col("a") + 1
    new_col_b = df.col("b") + 3
    result = df.assign(new_col_a, new_col_b)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [7, 8, 9]})
    pd.testing.assert_frame_equal(result_pd, expected)
    # check original df didn't change
    df_pd = interchange_to_pandas(df)
    df_pd = convert_dataframe_to_pandas_numpy(df_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(df_pd, expected)
