from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import PANDAS_VERSION
from tests.utils import mixed_dataframe_1


@pytest.mark.skipif(
    PANDAS_VERSION < (2, 0, 0),
    reason="no pyarrow support",
)
def test_schema(library: str) -> None:
    df = mixed_dataframe_1(library)
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
    assert isinstance(result["m"], namespace.Datetime)
    assert isinstance(result["n"], namespace.Datetime)
    if not (
        library.startswith("pandas")
        and tuple(int(v) for v in pd.__version__.split(".")) < (2, 0, 0)
    ):  # pragma: no cover (coverage bug?)
        # pandas non-nanosecond support only came in 2.0
        assert result["n"].time_unit == "ms"
    else:  # pragma: no cover
        pass
    assert result["n"].time_zone is None
    assert isinstance(result["o"], namespace.Datetime)
    if not (
        library.startswith("pandas")
        and tuple(int(v) for v in pd.__version__.split(".")) < (2, 0, 0)
    ):  # pragma: no cover (coverage bug?)
        # pandas non-nanosecond support only came in 2.0
        assert result["o"].time_unit == "us"
    else:  # pragma: no cover
        pass
    assert result["o"].time_zone is None
    if not (
        library.startswith("pandas")
        and tuple(int(v) for v in pd.__version__.split(".")) < (2, 0, 0)
    ):
        # pandas non-nanosecond support only came in 2.0 - before that, these would be 'float'
        assert isinstance(result["p"], namespace.Duration)
        assert result["p"].time_unit == "ms"
        assert isinstance(result["q"], namespace.Duration)
        assert result["q"].time_unit == "us"
    else:  # pragma: no cover
        pass
