from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_negative_powers(library: str) -> None:
    df = integer_dataframe_1(library)
    other = integer_dataframe_1(library) * -1
    with pytest.raises((ValueError, NotImplementedError)):
        df.__pow__(-1)
    with pytest.raises((ValueError, NotImplementedError)):
        df.__pow__(other)


def test_float_scalar_powers(library: str) -> None:
    df = integer_dataframe_1(library)
    other = 1.0
    result = df.__pow__(other)
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)


def test_float_powers(library: str) -> None:
    df = integer_dataframe_1(library)
    other = integer_dataframe_1(library) * 1.0
    if library == "polars-lazy":
        with pytest.raises(NotImplementedError):
            result = df.__pow__(other)
        return
    else:
        result = df.__pow__(other)
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [1.0, 4.0, 27.0], "b": [256.0, 3125.0, 46656.0]})
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)
