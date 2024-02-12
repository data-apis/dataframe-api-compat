from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import nan_dataframe_1


def test_column_fill_nan(library: BaseHandler) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.fill_nan(-1.0).rename("result"))
    expected = [1.0, 2.0, -1.0]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Float64)


def test_column_fill_nan_with_null(library: BaseHandler) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.fill_nan(ns.null).is_null().rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)
