from __future__ import annotations

import pandas as pd
import pytest
from packaging.version import Version
from packaging.version import parse

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


@pytest.mark.parametrize(
    ("func", "expected_data"),
    [
        ("cumulative_sum", [1, 3, 6]),
        ("cumulative_prod", [1, 2, 6]),
        ("cumulative_max", [1, 2, 3]),
        ("cumulative_min", [1, 1, 1]),
    ],
)
def test_cumulative_functions_column(
    library: BaseHandler,
    func: str,
    expected_data: list[float],
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    expected = pd.Series(expected_data, name="result")
    result = df.assign(getattr(ser, func)().rename("result"))

    if (
        parse(pd.__version__) < Version("2.0.0") and library.name == "pandas-nullable"
    ):  # pragma: no cover
        # Upstream bug
        result = result.cast({"result": ns.Int64()})

    compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)
