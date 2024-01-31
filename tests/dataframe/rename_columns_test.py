from __future__ import annotations

import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_rename(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.rename({"a": "c", "b": "e"})
    expected = {"c": [1, 2, 3], "e": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


def test_rename_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(
        TypeError,
        match="Expected Mapping, got: <class 'function'>",
    ):  # pragma: no cover
        # why is this not covered? bug in coverage?
        df.rename(lambda x: x.upper())  # type: ignore  # noqa: PGH003
