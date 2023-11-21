from __future__ import annotations

import pytest

from tests.utils import PANDAS_VERSION
from tests.utils import mixed_dataframe_1


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
@pytest.mark.skipif(
    PANDAS_VERSION < (2, 0, 0),
    reason="before pandas got non-nano support",
)
def test_is_dtype(library: str, dtype: str, expected: list[str]) -> None:
    df = mixed_dataframe_1(library).persist()
    namespace = df.__dataframe_namespace__()
    result = [i for i in df.column_names if namespace.is_dtype(df.schema[i], dtype)]
    assert result == expected
