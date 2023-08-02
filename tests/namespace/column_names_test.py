from __future__ import annotations

from typing import TYPE_CHECKING

from tests.utils import integer_series_1

if TYPE_CHECKING:
    import pytest


def test_column_names(library: str, request: pytest.FixtureRequest) -> None:
    # nameless column
    ser = integer_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]
    assert result.get_column_by_name("result").name == "result"

    # named column
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result"
    )
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]
    assert result.get_column_by_name("result").name == "result"

    # named column (different name)
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result2"
    )
    assert ser.name == "result2"
