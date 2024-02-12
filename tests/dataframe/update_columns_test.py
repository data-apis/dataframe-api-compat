from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_update_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    result = df.assign(col("a") + 1)
    expected = {"a": [2, 3, 4], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_update_multiple_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    result = df.assign(col("a") + 1, col("b") + 2)
    expected = {"a": [2, 3, 4], "b": [6, 7, 8]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
