from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import float_series_1
from tests.utils import float_series_2
from tests.utils import float_series_3
from tests.utils import float_series_4
from tests.utils import integer_series_1
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("ser_factory", "other_factory", "expected_values"),
    [
        (float_series_1, float_series_4, [False, False]),
        (float_series_2, float_series_4, [False, True]),
        (float_series_3, float_series_4, [True, False]),
    ],
)
def test_is_in(
    library: str,
    ser_factory: Callable[[str, pytest.FixtureRequest], Any],
    other_factory: Callable[[str, pytest.FixtureRequest], Any],
    expected_values: list[bool],
    request: pytest.FixtureRequest,
) -> None:
    other = other_factory(library, request)
    ser = ser_factory(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": (ser.is_in(other)).rename("result")}
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series(expected_values, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_is_in_raises(library: str, request: pytest.FixtureRequest) -> None:
    ser = float_series_1(library, request)
    other = integer_series_1(library, request)
    with pytest.raises(ValueError):
        ser.is_in(other)
