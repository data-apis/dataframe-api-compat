from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_name(library: str) -> None:
    df = integer_dataframe_1(library)
    name = df.get_column_by_name("a").name
    assert name == "a"
