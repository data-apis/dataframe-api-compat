from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import nan_dataframe_1


def test_column_is_nan(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.is_nan().rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)
