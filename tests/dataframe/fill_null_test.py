from __future__ import annotations

import pytest

from tests.utils import nan_dataframe_1
from tests.utils import null_dataframe_2


@pytest.mark.parametrize(
    "column_names",
    [
        ["a", "b"],
        None,
        ["a"],
        ["b"],
    ],
)
def test_fill_null(
    library: str,
    request: pytest.FixtureRequest,
    column_names: list[str] | None,
) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = null_dataframe_2(library, request)
    namespace = df.__dataframe_namespace__()
    result = df.fill_null(0, column_names=column_names)
    # todo: is there a way to test test this without if/then statements?
    if column_names is None or column_names == ["a", "b"]:
        if library == "polars":
            assert result.dataframe["a"][2] == 0.0
            assert result.dataframe["b"][2] == 0.0
        else:
            assert result.dataframe["a"][2] == 0.0
            assert result.dataframe["b"][2] == 0.0
    elif column_names == ["a"]:
        assert result.dataframe["a"][2] == 0.0
        assert namespace.is_null(result.dataframe["b"][2])
    elif column_names == ["b"]:
        assert namespace.is_null(result.dataframe["a"][2])
        assert result.dataframe["b"][2] == 0.0
    else:
        raise AssertionError()


def test_fill_null_noop(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = nan_dataframe_1(library)
    result = df.fill_null(0)
    # nan should not have changed!
    assert result.dataframe["a"][2] != result.dataframe["a"][2]
