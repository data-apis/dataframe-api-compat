from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1


def test_column_names(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    # nameless column
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
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
