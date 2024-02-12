from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_4


def test_group_by_size(library: BaseHandler) -> None:
    df = integer_dataframe_4(library)
    ns = df.__dataframe_namespace__()
    result = df.group_by("key").size()
    result = result.sort("key")
    expected = {"key": [1, 2], "size": [2, 2]}
    # TODO polars returns uint32. what do we standardise to?
    result = result.cast({"size": ns.Int64()})
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
