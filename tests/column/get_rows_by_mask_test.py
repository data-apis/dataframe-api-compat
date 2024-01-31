from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


def test_column_filter(library: str) -> None:
    df = integer_dataframe_1(library).persist()
    ser = df.get_column("a")
    mask = ser > 1
    ser = ser.filter(mask)
    result_pd = pd.Series(ser.to_array(), name="result")
    expected = pd.Series([2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.xfail(strict=False)
def test_column_take_by_mask_noop(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    mask = ser > 0
    ser = ser.filter(mask)
    result = df.assign(ser.rename("result"))
    compare_column_with_reference(result.col("result"), [1, 2, 3], dtype=pdx.Int64)
