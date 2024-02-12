from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import nan_dataframe_2
from tests.utils import null_dataframe_1


def test_is_null_1(library: BaseHandler) -> None:
    df = nan_dataframe_2(library)
    ns = df.__dataframe_namespace__()
    result = df.is_null()
    if library.name == "pandas-numpy":
        # nan and null are the same in pandas-numpy
        expected = {"a": [False, False, True]}
    else:
        expected = {"a": [False, False, False]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)


def test_is_null_2(library: BaseHandler) -> None:
    df = null_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.is_null()
    expected = {"a": [False, False, True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)
