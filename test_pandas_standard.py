import numpy as np
import pytest
import pandas as pd
from pandas_standard import PandasDataFrame, PandasColumn

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

def test_get_column_by_name():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    result = df.get_column_by_name('a')
    np.testing.assert_array_equal(np.array([1,2,3]), np.from_dlpack(result))

def test_get_columns_by_name():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    result = df.get_columns_by_name(['b']).dataframe
    expected = pd.DataFrame({'b': [4,5,6]})
    pd.testing.assert_frame_equal(result, expected)

def test_get_rows():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    result = df.get_rows([0, 2]).dataframe
    expected = pd.DataFrame({'a': [1,3], 'b': [4,6]}, index=[0, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_slice_rows():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3,4,5,6, 7], 'b': [7, 6,5,4,3,2,1]}))
    result = df.slice_rows(2, 7, 2).dataframe
    expected = pd.DataFrame({'a': [3, 5, 7], 'b': [5, 3, 1]}, index=[2, 4, 6])
    pd.testing.assert_frame_equal(result, expected)

def test_get_rows_by_mask():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    mask = PandasColumn(pd.Series([True, False, True]))
    result = df.get_rows_by_mask(mask).dataframe
    expected = pd.DataFrame({'a': [1, 3], 'b': [4, 6]}, index=[0, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_insert():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    new_col = PandasColumn(pd.Series([7,8,9]))
    result = df.insert(1, 'c', new_col)
    expected = pd.DataFrame({'a': [1, 2,3], 'c': [7,8,9], 'b': [4, 5,6]})
    pd.testing.assert_frame_equal(result, expected)

def test_drop_column():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    result = df.drop_column('a').dataframe
    expected = pd.DataFrame({'b': [4, 5,6]})
    pd.testing.assert_frame_equal(result, expected)

def test_set_column():
    # I'm hoping to get rid of this one, so holding off for now... 
    ...

def test_rename_columns():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    result = df.rename_columns({'a': 'c', 'b': 'e'}).dataframe
    expected = pd.DataFrame({'c': [1,2,3], 'e': [4,5,6]})
    pd.testing.assert_frame_equal(result, expected)

def test_get_column_names():
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    result = df.get_column_names()
    assert [name for name in result] == ['a', 'b']

def test_groupby():
    ...
