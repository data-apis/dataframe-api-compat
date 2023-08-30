from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_rename(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
    result = ser.rename("new_name")
    assert result.name == "new_name"
