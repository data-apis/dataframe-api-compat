from __future__ import annotations

import sys
from typing import Any

LIBRARIES = {
    (3, 8): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 9): ["pandas-nullable", "pandas-numpy"],
    (3, 10): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 11): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 12): ["polars-lazy"],
}


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        metafunc.parametrize(
            "library",
            LIBRARIES[sys.version_info[:2]],
        )


import modin.pandas as pd

class NameSpaceCustom:
    def assert_series_equal(df1, df2, check_exact=False):
        from modin.pandas.test.utils import df_equals
        df_equals(df1, df2, check_exact=check_exact)

    def assert_frame_equal(df1, df2):
        from modin.pandas.test.utils import df_equals
        df_equals(df1, df2)

pd.testing = NameSpaceCustom

pd.__version__ = "0.25.0"
