from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_to_standard_compliant_dataframe
from tests.utils import integer_dataframe_1


def test_name(library: str) -> None:
    df = integer_dataframe_1(library).persist()
    name = df.col("a").name
    assert name == "a"


def test_pandas_name_if_0_named_column() -> None:
    df = convert_to_standard_compliant_dataframe(pd.DataFrame({0: [1, 2, 3]}))
    assert df.column_names == [0]  # type: ignore[comparison-overlap]
    assert [col.name for col in df.iter_columns()] == [0]  # type: ignore[comparison-overlap]


@pytest.mark.skipif(
    tuple(int(v) for v in pd.__version__.split(".")) < (2, 1, 0),
    reason="before consoritum standard",
)
def test_invalid_name_pandas() -> None:
    with pytest.raises(ValueError):
        pd.Series([1, 2, 3], name=0).__column_consortium_standard__()
