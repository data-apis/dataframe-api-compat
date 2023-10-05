from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_root_names(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col

    assert col("a").root_names == ["a"]
    assert col("b").root_names == ["b"]
    assert col("b").rename("c").root_names == ["b"]
    assert (col("b") + col("a")).root_names == ["a", "b"]
    assert (col("b") + col("a") + col("a")).root_names == ["a", "b"]
    assert namespace.any_rowwise("a", "b").root_names == ["a", "b"]
