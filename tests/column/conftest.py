
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
