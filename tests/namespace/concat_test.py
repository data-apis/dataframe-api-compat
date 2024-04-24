from __future__ import annotations

from typing import Any

import pytest

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import integer_dataframe_4


def test_concat(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    ns = df1.__dataframe_namespace__()
    result = ns.concat([df1, df2])
    expected = {"a": [1, 2, 3, 1, 2, 4], "b": [4, 5, 6, 4, 2, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_concat_mismatch(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library).persist()
    df2 = integer_dataframe_4(library).persist()
    ns = df1.__dataframe_namespace__()
    exceptions: tuple[Any, ...] = (ValueError,)
    if library.name == "polars-lazy":
        import polars as pl

        exceptions = (ValueError, pl.exceptions.ShapeError)
    # TODO check the error
    with pytest.raises(exceptions):
        _ = ns.concat([df1, df2]).persist()
