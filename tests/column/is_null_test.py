from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import nan_dataframe_1
from tests.utils import null_dataframe_1


def test_column_is_null_1(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.is_null().rename("result"))
    if library.name == "pandas-numpy":
        expected = [False, False, True]
    else:
        expected = [False, False, False]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)


def test_column_is_null_2(library: BaseHandler) -> None:
    df = null_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.is_null().rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)
