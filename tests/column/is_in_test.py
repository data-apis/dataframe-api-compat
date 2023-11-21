from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import pytest

from tests.utils import float_dataframe_1
from tests.utils import float_dataframe_2
from tests.utils import float_dataframe_3
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("df_factory", "expected_values"),
    [
        (float_dataframe_1, [False, True]),
        (float_dataframe_2, [True, False]),
        (float_dataframe_3, [True, False]),
    ],
)
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_is_in(
    library: str,
    df_factory: Callable[[str], Any],
    expected_values: list[bool],
) -> None:
    df = df_factory(library).persist()
    ser = df.col("a")
    other = ser + 1
    result = df.assign(ser.is_in(other).rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series(expected_values, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("df_factory", "expected_values"),
    [
        (float_dataframe_1, [False, True]),
        (float_dataframe_2, [True, False]),
        (float_dataframe_3, [True, False]),
    ],
)
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_expr_is_in(
    library: str,
    df_factory: Callable[[str], Any],
    expected_values: list[bool],
) -> None:
    df = df_factory(library)
    col = df.col
    ser = col("a")
    other = ser + 1
    result = df.assign(ser.is_in(other).rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series(expected_values, name="result")
    pd.testing.assert_series_equal(result_pd, expected)
