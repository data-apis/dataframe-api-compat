from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION


@pytest.mark.skipif(
    POLARS_VERSION < (0, 19, 0) or PANDAS_VERSION < (2, 1, 0),
    reason="before consortium standard in polars/pandas",
)
def test_convert_to_std_column() -> None:
    s = pl.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean()) == 2
    s = pl.Series("bob", [1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean()) == 2
    s = pd.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean()) == 2
    s = pd.Series([1, 2, 3], name="alice").__column_consortium_standard__()
    assert float(s.mean()) == 2
