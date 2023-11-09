from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from tests.utils import interchange_to_pandas
from tests.utils import temporal_dataframe_1


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("1day", [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]),
    ],
)
def test_floor(library: str, freq: str, expected: list[datetime]) -> None:
    df = temporal_dataframe_1(library)
    col = df.col
    result = df.assign(col("a").floor(freq).rename("result")).select("result").persist()  # type: ignore[attr-defined]
    # TODO check the resolution
    result = interchange_to_pandas(result)["result"].astype("datetime64[ns]")
    expected = pd.Series(expected, name="result")
    pd.testing.assert_series_equal(result, expected)
