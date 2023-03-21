import pytest
import pandas as pd
from pandas_standard import PandasDataFrame

@pytest.mark.parametrize(
    ('reduction', 'expected_data'),
    [
        ('any', {'a': [True], 'b': [True]}),
        ('all', {'a': [False], 'b': [True]}),
    ]
)
def test_reductions(reduction, expected_data):
    df = pd.DataFrame({'a': [True, True, False], 'b': [True, True, True]})
    dfstd = PandasDataFrame(df)
    result = getattr(dfstd, reduction)().dataframe
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result, expected)

@pytest.mark.parametrize(
    ('comparison', 'expected_data'),
    [
        ('__eq__', {'a': [True, True, False], 'b': [True, False, True]}),
        ('__ne__', {'a': [False, False, True], 'b': [False, True, False]}),
        ('__ge__', {'a': [True, True, False], 'b': [True, True, True]}),
        ('__gt__', {'a': [False, False, False], 'b': [False, True, False]}),
        ('__le__', {'a': [True, True, True], 'b': [True, False, True]}),
        ('__lt__', {'a': [False, False, True], 'b': [False, False, False]}),
        ('__add__', {'a': [2, 4, 7], 'b': [8, 7, 12]}),
        ('__sub__', {'a': [0, 0, -1], 'b': [0, 3, 0]}),
        ('__mul__', {'a': [1, 4, 12], 'b': [16, 10, 36]}),
        ('__truediv__', {'a': [1, 1, .75], 'b': [1, 2.5, 1]}),
        ('__floordiv__', {'a': [1, 1, 0], 'b': [1, 2, 1]}),
        ('__pow__', {'a': [1, 4, 81], 'b': [256, 25, 46656]}),
        ('__mod__', {'a': [0, 0, 3], 'b': [0, 1, 0]}),
    ]
)
def test_comparisons(comparison, expected_data):
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    other = PandasDataFrame(pd.DataFrame({'a': [1,2,4], 'b': [4,2,6]}))
    result = getattr(df, comparison)(other).dataframe
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result, expected)

def test_divmod():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    other = PandasDataFrame(pd.DataFrame({'a': [1,2,4], 'b': [4,2,6]}))
    result_quotient, result_remainder = df.__divmod__(other)
    expected_quotient = pd.DataFrame({'a': [1, 1, 0], 'b': [1, 2, 1]})
    expected_remainder = pd.DataFrame({'a': [0, 0, 3], 'b': [0, 1, 0]})
    pd.testing.assert_frame_equal(result_quotient.dataframe, expected_quotient)
    pd.testing.assert_frame_equal(result_remainder.dataframe, expected_remainder)
