from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_output_name(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col

    assert col("a").output_name == "a"
    assert col("b").output_name == "b"
    assert col("b").rename("c").output_name == "c"
    assert (col("b") + col("a")).output_name == "b"
    assert (col("b") + col("a") + col("a")).output_name == "b"
    assert namespace.any_rowwise(["a", "b"]).output_name == "any"
