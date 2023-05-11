Work-in-progress draft of what the [DataFrame Standard](https://data-apis.org/dataframe-api/draft/index.html)
might look like for pandas and polars.

Example
-------

Say you want to write a function which takes a DataFrame `df` and want to split it into
two sub-dataframes: one with rows the value of column `'x'` is `True`, and another with rows when the value
of column `'x'` is `False`. In `pandas` this would be:
```python
def split_df(df):
    return df[df['x']], df[!df['x']]
```
whereas in polars, this would be:
```python
def split_df(df):
    return df.filter(pl.col('x')==True), df.filter(pl.col('x')==False)
```

Using the DataFrame Standard, however, it would be possible to write portable code which would work
for both libraries, allowing library authours to write DataFrame-agnostic code!
```python
def split_df(df):
    # Note: this isn't implemented yet and is just for illustrative purposes
    df_standard = df.__dataframe_standard__()
    mask = df_standard.get_column_by_name('x') == True
    df0 = df_standard.get_rows_by_mask(mask)
    df1 = df_standard.get_rows_by_mask(~mask)
    return df0.dataframe, df1.dataframe
```

How to try this out
-------------------

The objective is to get the point where
```python
df.__dataframe_standard__()
```
would return a Standard-compliant DataFrame object.

We are not there yet - to try out the Standard with this MVP, you can use the
`pandas_standard.PandasDataFrame` class:
```python
from pandas_standard import PandasDataFrame

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

df_standard = PandasDataFrame(df)  # standard-compliant DataFrame
```

Installation
------------
```
pip install git+https://github.com/MarcoGorelli/impl-dataframe-api
```

Testing
-------
```
pytest --cov=pandas_standard
```

Linting
-------
```
pre-commit run --all-files
```
