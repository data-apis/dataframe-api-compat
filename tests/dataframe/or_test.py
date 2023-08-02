from __future__ import annotations

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import bool_dataframe_4
from tests.utils import convert_dataframe_to_pandas_numpy


def test_or(library: str) -> None:
    df = bool_dataframe_1(library)
    other = bool_dataframe_4(library)
    result = df | other
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [True, True, False], "b": [True, True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_or_with_scalar(library: str) -> None:
    df = bool_dataframe_1(library)
    other = True
    result = df | other
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [True, True, True], "b": [True, True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
