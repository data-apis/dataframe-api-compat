from __future__ import annotations

import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import mixed_dataframe_1
from tests.utils import pandas_version


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("integral", ["a", "b", "c", "d", "e", "f", "g", "h"]),
        ("signed integer", ["a", "b", "c", "d"]),
        ("unsigned integer", ["e", "f", "g", "h"]),
        ("floating", ["i", "j"]),
        ("bool", ["k"]),
        ("string", ["l"]),
        (("string", "integral"), ["a", "b", "c", "d", "e", "f", "g", "h", "l"]),
        (("string", "unsigned integer"), ["e", "f", "g", "h", "l"]),
    ],
)
def test_is_dtype(library: BaseHandler, dtype: str, expected: list[str]) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable") and pandas_version() < Version(
        "2.0.0",
    ):  # pragma: no cover
        pytest.skip(reason="pandas before non-nano")
    df = mixed_dataframe_1(library).persist()
    namespace = df.__dataframe_namespace__()
    result = [i for i in df.column_names if namespace.is_dtype(df.schema[i], dtype)]
    assert result == expected
