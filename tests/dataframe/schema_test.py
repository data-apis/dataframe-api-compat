from __future__ import annotations

from typing import Any
from typing import Callable

import pytest

from tests.utils import mixed_dataframe_1


@pytest.mark.parametrize("maybe_collect", [lambda x: x, lambda x: x.persist()])
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
    assert isinstance(result["c"], namespace.Int16)
    assert isinstance(result["d"], namespace.Int8)
    assert isinstance(result["e"], namespace.UInt64)
    assert isinstance(result["f"], namespace.UInt32)
    assert isinstance(result["g"], namespace.UInt16)
    assert isinstance(result["h"], namespace.UInt8)
    assert isinstance(result["i"], namespace.Float64)
    assert isinstance(result["j"], namespace.Float32)
    assert isinstance(result["k"], namespace.Bool)
    assert isinstance(result["l"], namespace.String)
    assert isinstance(result["m"], namespace.Date)
    assert isinstance(result["n"], namespace.Datetime)
    assert result["n"].time_unit == "ms"
    assert result["n"].time_zone is None
    assert isinstance(result["o"], namespace.Datetime)
    assert result["o"].time_unit == "us"
    assert result["o"].time_zone is None
    assert isinstance(result["p"], namespace.Duration)
    assert result["p"].time_unit == "ms"
    assert isinstance(result["q"], namespace.Duration)
    assert result["q"].time_unit == "us"
