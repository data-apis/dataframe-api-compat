from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import nan_dataframe_1


def test_dataframe_is_nan(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.is_nan()
    expected = {"a": [False, False, True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)
