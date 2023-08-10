from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("min", 1),
        ("max", 3),
        ("sum", 6),
        ("prod", 6),
        ("median", 2.0),
        ("mean", 2.0),
        ("std", 1.0),
        ("var", 1.0),
    ],
)
def test_column_reductions(
    library: str, reduction: str, expected: float, request: pytest.FixtureRequest
) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    if library == "polars-lazy":
        with pytest.raises(NotImplementedError):
            result = getattr(ser, reduction)()
        return
    result = getattr(ser, reduction)()
    assert result == expected
