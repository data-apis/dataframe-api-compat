from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_is_null(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    null = namespace.null
    assert namespace.is_null(null)
    assert not namespace.is_null(float("nan"))
    assert not namespace.is_null(0)
