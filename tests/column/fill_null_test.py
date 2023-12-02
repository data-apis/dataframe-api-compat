from __future__ import annotations

from tests.utils import nan_dataframe_1
from tests.utils import null_dataframe_2


def test_fill_null_column(library: str) -> None:
    df = null_dataframe_2(library)
    ser = df.col("a")
    result = df.assign(ser.fill_null(0).rename("result")).col("result")
    assert float(result.get_value(2).persist()) == 0.0  # type:ignore[arg-type]
    assert float(result.get_value(1).persist()) != 0.0  # type:ignore[arg-type]
    assert float(result.get_value(0).persist()) != 0.0  # type:ignore[arg-type]


def test_fill_null_noop_column(library: str) -> None:
    df = nan_dataframe_1(library)
    ser = df.col("a")
    result = df.assign(ser.fill_null(0).rename("result")).persist().col("result")
    if library != "pandas-numpy":
        # nan should not have changed!
        assert float(result.get_value(2)) != float(  # type: ignore[arg-type]
            result.get_value(2),  # type: ignore[arg-type]
        )
    else:
        # nan was filled with 0
        assert float(result.get_value(2)) == 0  # type: ignore[arg-type]
    assert float(result.get_value(1)) != 0.0  # type: ignore[arg-type]
    assert float(result.get_value(0)) != 0.0  # type: ignore[arg-type]
