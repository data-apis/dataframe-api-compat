from __future__ import annotations

import polars as pl
import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import integer_dataframe_4


def test_concat(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    ns = df1.__dataframe_namespace__()
    result = ns.concat([df1, df2])
    expected = {"a": [1, 2, 3, 1, 2, 4], "b": [4, 5, 6, 4, 2, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_concat_mismatch(library: str) -> None:
    df1 = integer_dataframe_1(library).persist()
    df2 = integer_dataframe_4(library).persist()
    ns = df1.__dataframe_namespace__()
    # TODO check the error
    with pytest.raises((ValueError, pl.exceptions.ShapeError)):
        _ = ns.concat([df1, df2]).persist()
