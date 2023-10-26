from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_6
from tests.utils import interchange_to_pandas


def test_column_sorted_indices_ascending(library: str) -> None:
    df = integer_dataframe_6(library)
    sorted_indices = df.col("b").sorted_indices()
    result = df.assign(sorted_indices.rename("result"))
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected_1 = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 1, 2],
            "result": [3, 4, 2, 0, 1],
        },
    )
    expected_2 = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 1, 2],
            "result": [3, 4, 2, 1, 0],
        },
    )
    if library in ("polars", "polars-lazy"):
        result_pd["result"] = result_pd["result"].astype("int64")
    try:
        pd.testing.assert_frame_equal(result_pd, expected_1)
    except AssertionError:  # pragma: no cover
        # order isn't determinist, so try both
        pd.testing.assert_frame_equal(result_pd, expected_2)


def test_column_sorted_indices_descending(library: str) -> None:
    df = integer_dataframe_6(library)
    sorted_indices = df.col("b").sorted_indices(ascending=False)
    result = df.assign(sorted_indices.rename("result"))
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected_1 = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 1, 2],
            "result": [1, 0, 2, 4, 3],
        },
    )
    expected_2 = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [4, 4, 3, 1, 2],
            "result": [0, 1, 2, 4, 3],
        },
    )
    if library in ("polars", "polars-lazy"):
        result_pd["result"] = result_pd["result"].astype("int64")
    try:
        pd.testing.assert_frame_equal(result_pd, expected_1)
    except AssertionError:
        # order isn't determinist, so try both
        pd.testing.assert_frame_equal(result_pd, expected_2)
