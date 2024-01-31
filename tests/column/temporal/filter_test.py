from tests.utils import compare_dataframe_with_reference
from tests.utils import temporal_dataframe_1


def test_filter_w_date(library: str) -> None:
    df = temporal_dataframe_1(library).select("a", "index")
    pdx = df.__dataframe_namespace__()
    result = df.filter(pdx.col("a") > pdx.date(2020, 1, 2)).select("index")
    compare_dataframe_with_reference(result, {"index": [1, 2]}, dtype=pdx.Int64)
