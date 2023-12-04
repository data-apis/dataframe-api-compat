from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


def test_expression_take(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    indices = df.col("a") - 1
    result = df.assign(ser.take(indices).rename("result")).select("result")
    compare_column_with_reference(result.col("result"), [1, 2, 3], dtype=ns.Int64)
