from __future__ import annotations

from tests.utils import bool_dataframe_1


def test_expr_any(library: str) -> None:
    df = bool_dataframe_1(library).collect()
    result = df.col("a").any()  # hack to test value
    assert bool(result)


def test_expr_all(library: str) -> None:
    df = bool_dataframe_1(library).collect()
    result = df.col("a").all()  # hack to test value
    assert not bool(result)
