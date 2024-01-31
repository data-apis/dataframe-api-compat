from __future__ import annotations

from datetime import datetime

import pytest

from tests.utils import compare_column_with_reference
from tests.utils import temporal_dataframe_1


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("1day", [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]),
    ],
)
def test_floor(library: str, freq: str, expected: list[datetime]) -> None:
    df = temporal_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.assign(pdx.col("a").floor(freq).rename("result")).select("result").persist()  # type: ignore[attr-defined]
    # TODO check the resolution
    compare_column_with_reference(
        result.get_column("result"),
        expected,
        dtype=pdx.Datetime,
    )
