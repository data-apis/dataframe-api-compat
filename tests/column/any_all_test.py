from __future__ import annotations

import pytest

from tests.utils import bool_dataframe_1


def test_expr_any(library: str) -> None:
    df = bool_dataframe_1(library)
    with pytest.raises(ValueError):
        bool(df.get_column("a").any())
    df = df.persist()
    result = df.get_column("a").any()
    assert bool(result)


def test_expr_all(library: str) -> None:
    df = bool_dataframe_1(library).persist()
    pdx = df.__dataframe_namespace__()
    result = pdx.col("a").all()
    assert bool(result)
