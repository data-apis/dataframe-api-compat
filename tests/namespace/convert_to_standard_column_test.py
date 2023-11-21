from __future__ import annotations

import sys

import pandas as pd
import polars as pl


def test_convert_to_std_column() -> None:
    s = pl.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
    s = pl.Series("bob", [1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
    if (
        sys.version_info < (3, 9)
        or sys.version_info >= (3, 12)
        or tuple(int(v) for v in pd.__version__.split(".")) < (2, 1, 0)
    ):
        # pandas doesn't support 3.8
        return
    s = pd.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
    s = pd.Series([1, 2, 3], name="alice").__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
