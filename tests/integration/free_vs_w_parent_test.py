import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from tests.utils import integer_dataframe_1


def test_free_vs_w_parent(library: str) -> None:
    df1 = integer_dataframe_1(library)
    namespace = df1.__dataframe_namespace__()
    free_ser1 = namespace.column_from_1d_array(  # type: ignore[call-arg]
        np.array([1, 2, 3], dtype="int64"),
        name="preds",
    )
    free_ser2 = namespace.column_from_1d_array(  # type: ignore[call-arg]
        np.array([4, 5, 6], dtype="int64"),
        name="preds",
    )

    result = free_ser1 + free_ser2
    if library == "polars-lazy":
        assert_series_equal(
            pl.select(result.column)["preds"],
            pl.Series("preds", [5, 7, 9], dtype=pl.Int64()),
        )

    assert namespace.is_dtype(result.dtype, "integral")
