from __future__ import annotations

import sys

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1


def test_name(library: str) -> None:
    df = integer_dataframe_1(library).persist()
    name = df.col("a").name
    assert name == "a"


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="pandas doesn't support 3.8",
)
def test_invalid_name_pandas() -> None:
    with pytest.raises(ValueError):
        pd.Series([1, 2, 3], name=0).__column_consortium_standard__()
