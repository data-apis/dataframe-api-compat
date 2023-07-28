from tests.utils import integer_dataframe_4
import pytest


def test_groupby_invalid_any_all(library: str) -> None:
    df = integer_dataframe_4(library)
    with pytest.raises(Exception):
        df.groupby(["key"]).any()
    with pytest.raises(Exception):
        df.groupby(["key"]).all()
