from __future__ import annotations

import pytest

from tests.utils import bool_dataframe_1


def test_expr_any(library: str) -> None:
    df = bool_dataframe_1(library)
    with pytest.raises(ValueError):
        bool(df.col("a").any())
    df = df.collect()
    result = df.col("a").any()
    assert bool(result)


def test_expr_all(library: str) -> None:
    df = bool_dataframe_1(library).collect()
    result = df.col("a").all()
    assert not bool(result)
