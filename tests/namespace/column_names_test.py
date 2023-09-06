from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1


def test_column_names(library: str, request: pytest.FixtureRequest) -> None:
    return None
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    # nameless column
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
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
