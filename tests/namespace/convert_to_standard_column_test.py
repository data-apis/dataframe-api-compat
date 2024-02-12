from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from packaging.version import Version

from tests.utils import pandas_version
from tests.utils import polars_version


@pytest.mark.skipif(
    Version("0.19.0") > polars_version() or Version("2.1.0") > pandas_version(),
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
