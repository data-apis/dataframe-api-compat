from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tests.utils import float_dataframe_1
from tests.utils import float_dataframe_2
from tests.utils import float_dataframe_3
from tests.utils import integer_dataframe_1
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
def test_is_in(
    library: str,
    df_factory: Callable[[str, pytest.FixtureRequest], Any],
    expected_values: list[bool],
    request: pytest.FixtureRequest,
) -> None:
    df = df_factory(library, request)
    ser = df.get_column_by_name("a")
    other = ser + 1
    result = df.insert(0, "result", ser.is_in(other))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series(expected_values, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_is_in_raises(library: str) -> None:
    ser = integer_dataframe_1(library).get_column_by_name("a")
    other = ser * 1.0
    with pytest.raises(ValueError):
        ser.is_in(other)
