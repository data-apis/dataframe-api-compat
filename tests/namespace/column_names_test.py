from __future__ import annotations

from typing import TYPE_CHECKING

from tests.utils import integer_dataframe_1

if TYPE_CHECKING:
    import pytest


def test_column_names(library: str, request: pytest.FixtureRequest) -> None:
    # nameless column
    df = integer_dataframe_1(library).collect()
    namespace = df.__dataframe_namespace__()
    ser = df.get_column_by_name("a")
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]

    # named column
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result"
    )
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]

    # named column (different name)
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result2"
    )
    assert ser.name == "result2"
