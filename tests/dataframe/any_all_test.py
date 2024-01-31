from __future__ import annotations

import pytest

from tests.utils import bool_dataframe_1
from tests.utils import bool_dataframe_3
from tests.utils import compare_dataframe_with_reference


@pytest.mark.parametrize(
    ("reduction", "expected_data"),
    [
        ("any", {"a": [True], "b": [True]}),
        ("all", {"a": [False], "b": [True]}),
    ],
)
def test_reductions(
    library: str,
    reduction: str,
    expected_data: dict[str, object],
) -> None:
    df = bool_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = getattr(df, reduction)()
    compare_dataframe_with_reference(result, expected_data, dtype=pdx.Bool)  # type: ignore[arg-type]


def test_any(library: str) -> None:
    df = bool_dataframe_3(library)
    pdx = df.__dataframe_namespace__()
    result = df.any()
    expected = {"a": [False], "b": [True], "c": [True]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Bool)


def test_all(library: str) -> None:
    df = bool_dataframe_3(library)
    pdx = df.__dataframe_namespace__()
    result = df.all()
    expected = {"a": [False], "b": [False], "c": [True]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Bool)
