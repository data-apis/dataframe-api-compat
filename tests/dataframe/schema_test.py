from __future__ import annotations

from typing import Any, Callable

import pytest

from tests.utils import mixed_dataframe_1


@pytest.mark.parametrize("maybe_collect", [lambda x: x, lambda x: x.collect()])
def test_schema(library: str, maybe_collect: Callable[[Any], Any]) -> None:
    df = maybe_collect(mixed_dataframe_1(library))
    namespace = df.__dataframe_namespace__()
    result = df.schema
    assert list(result.keys()) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
    ]
    assert isinstance(result["a"], namespace.Int64)
    assert isinstance(result["b"], namespace.Int32)
