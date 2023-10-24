from __future__ import annotations

import pandas as pd
import polars as pl


def test_convert_to_std_column() -> None:
    s = pl.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean()) == 2
    s = pl.Series("bob", [1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean()) == 2
    s = pd.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean()) == 2
    s = pd.Series([1, 2, 3], name="alice").__column_consortium_standard__()
    assert float(s.mean()) == 2
