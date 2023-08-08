from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_5
from tests.utils import interchange_to_pandas


def test_sorted_indices(
    library: str,
) -> None:
    df = integer_dataframe_5(library)
    sorted_indices = df.sorted_indices(keys=["a", "b"])
    result = df.get_rows(sorted_indices)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 1], "b": [3, 4]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_sorted_indices_descending(
    library: str,
) -> None:
    df = integer_dataframe_5(library)
    sorted_indices = df.sorted_indices(keys=["a", "b"], ascending=False)
    result = df.get_rows(sorted_indices)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 1], "b": [4, 3]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_sorted_indices_with_insert(library: str) -> None:
    df = integer_dataframe_5(library)
    result = df.insert(0, "idx", df.sorted_indices())
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    result_pd["idx"] = result_pd["idx"].astype("int64")
    expected = pd.DataFrame({"idx": [1, 0], "a": [1, 1], "b": [4, 3]})
    pd.testing.assert_frame_equal(result_pd, expected)
