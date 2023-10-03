from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_get_column_names(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.column_names
    assert list(result) == ["a", "b"]
