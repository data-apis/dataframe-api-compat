from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import bool_dataframe_1
from tests.utils import compare_dataframe_with_reference


def test_and_with_scalar(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    other = True
    result = df & other
    expected = {"a": [True, True, False], "b": [True, True, True]}
    compare_dataframe_with_reference(result, expected, ns.Bool)


def test_rand_with_scalar(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    other = True
    result = other & df
    expected = {"a": [True, True, False], "b": [True, True, True]}
    compare_dataframe_with_reference(result, expected, ns.Bool)
