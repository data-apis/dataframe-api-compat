from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_name(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    name = namespace.col("a").name
    assert name == "a"
