from __future__ import annotations

import pandas as pd

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


def test_column_filter(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ser = df.col("a")
    mask = ser > 1
    ser = ser.filter(mask).persist()
    result_pd = pd.Series(ser.to_array(), name="result")
    expected = pd.Series([2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_take_by_mask_noop(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    mask = ser > 0
    ser = ser.filter(mask)
    result = df.assign(ser.rename("result"))
    compare_column_with_reference(result.col("result"), [1, 2, 3], dtype=ns.Int64)
