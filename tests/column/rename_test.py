from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_rename(library: str) -> None:
    integer_dataframe_1(library).persist()
    ser = pdx.col("a")
    result = ser.rename("new_name")
    assert result.name == "new_name"
