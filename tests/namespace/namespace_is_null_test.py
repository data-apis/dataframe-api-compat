from __future__ import annotations

from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2


def test_is_null(library: str) -> None:
    df = integer_dataframe_1(library)
    other = integer_dataframe_2(library)
    namespace = df.__dataframe_namespace__()
    namespace_other = other.__dataframe_namespace__()
    null = namespace.null
    assert namespace_other.is_null(null)
    assert not namespace_other.is_null(float("nan"))
    assert not namespace_other.is_null(0)
