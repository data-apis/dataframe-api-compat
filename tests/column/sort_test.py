from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_6
from tests.utils import interchange_to_pandas


def test_column_sort_ascending(library: str) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    sorted = df.get_column_by_name("b").sort().rename("c")
    result = df.insert_column(sorted)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 1, 2],
            "c": [1, 2, 3, 4, 4],
        }
    )
    pd.testing.assert_frame_equal(result_pd, expected)


def test_column_sort_descending(library: str) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    sorted = df.get_column_by_name("b").sort(ascending=False).rename("c")
    result = df.insert_column(sorted)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 1, 2],
            "c": [4, 4, 3, 2, 1],
        }
    )
    pd.testing.assert_frame_equal(result_pd, expected)
