from typing import Callable, Any
import pytest
import pandas as pd
from tests.utils import float_series_1
from tests.utils import float_series_2
from tests.utils import float_series_3
from tests.utils import float_series_4
from tests.utils import integer_series_1
from tests.utils import convert_series_to_pandas_numpy


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
    ser_factory: Callable[[str], Any],
    other_factory: Callable[[str], Any],
    expected_values: list[bool],
) -> None:
    other = other_factory(library)
    ser = ser_factory(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.is_in(other)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series(expected_values, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_is_in_raises(library: str) -> None:
    ser = float_series_1(library)
    other = integer_series_1(library)
    with pytest.raises(ValueError):
        ser.is_in(other)
