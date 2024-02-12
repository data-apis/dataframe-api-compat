from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_filter(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    mask = df.col("a") % 2 == 1
    result = df.filter(mask)
    expected = {"a": [1, 3], "b": [4, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
