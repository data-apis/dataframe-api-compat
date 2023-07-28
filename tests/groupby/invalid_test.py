from tests.utils import integer_dataframe_1
import pytest


def test_groupby_invalid(library: str) -> None:
    df = integer_dataframe_1(library).get_columns_by_name(["a"])
    with pytest.raises((KeyError, TypeError)):
        df.groupby(0)
    with pytest.raises((KeyError, TypeError)):
        df.groupby("0")
    with pytest.raises((KeyError, TypeError)):
        df.groupby(["b"])