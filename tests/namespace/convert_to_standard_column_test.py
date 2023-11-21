from __future__ import annotations

import pandas as pd
import polars as pl
import pytest


@pytest.mark.skipif(
    tuple(int(v) for v in pl.__version__.split(".")) < (0, 18, 0)
    or tuple(int(v) for v in pd.__version__.split(".")) < (2, 1, 0),
    reason="before consortium standard in polars/pandas",
)
def test_convert_to_std_column() -> None:
    s = pl.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
    s = pl.Series("bob", [1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
    s = pd.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
    s = pd.Series([1, 2, 3], name="alice").__column_consortium_standard__()
    assert float(s.mean().persist()) == 2
