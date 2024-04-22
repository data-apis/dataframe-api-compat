from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_take(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    df = df.assign((df.col("a") - 1).sort(ascending=False).rename("result"))
    result = df.take(df.col("result"))
    expected = {"a": [3, 2, 1], "b": [6, 5, 4], "result": [0, 1, 2]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
