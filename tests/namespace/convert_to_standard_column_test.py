from __future__ import annotations

import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import pandas_version
from tests.utils import polars_version


def test_convert_to_std_column(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable"):
        if pandas_version() < Version("2.1.0"):
            pytest.skip(reason="before consortium standard in pandas")
        import pandas as pd

        s = pd.Series([1, 2, 3]).__column_consortium_standard__()
        assert float(s.mean()) == 2
        s = pd.Series([1, 2, 3], name="alice").__column_consortium_standard__()
        assert float(s.mean()) == 2
    elif library.name == "polars-lazy":
        if polars_version() < Version("0.19.0"):
            pytest.skip(reason="before consortium standard in polars")
        import polars as pl

        s = pl.Series([1, 2, 3]).__column_consortium_standard__()
        assert float(s.mean()) == 2
        s = pl.Series("bob", [1, 2, 3]).__column_consortium_standard__()
        assert float(s.mean()) == 2
