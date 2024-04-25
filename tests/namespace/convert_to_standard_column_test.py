from __future__ import annotations

import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import pandas_version
from tests.utils import polars_version


def test_convert_to_std_column(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable"):
        if pandas_version() < Version("2.1.0"):  # pragma: no cover
            pytest.skip(reason="before consortium standard in pandas")
        import pandas as pd

        ser = pd.Series([1, 2, 3]).__column_consortium_standard__()
        ser_with_name = pd.Series(
            [1, 2, 3],
            name="alice",
        ).__column_consortium_standard__()
    elif library.name == "polars-lazy":
        if polars_version() < Version("0.19.0"):  # pragma: no cover
            pytest.skip(reason="before consortium standard in polars")
        import polars as pl

        ser = pl.Series([1, 2, 3]).__column_consortium_standard__()
        ser_with_name = pl.Series("bob", [1, 2, 3]).__column_consortium_standard__()
    elif library.name == "modin":
        import modin.pandas as pd

        ser = pd.Series([1, 2, 3]).__column_consortium_standard__()
        ser_with_name = pd.Series(
            [1, 2, 3],
            name="alice",
        ).__column_consortium_standard__()
    else:  # pragma: no cover
        msg = f"Not supported library: {library}"
        raise AssertionError(msg)

    assert float(ser.mean()) == 2
    assert float(ser_with_name.mean()) == 2
