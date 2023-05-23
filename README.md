# DataFrame Standard POC

<h1 align="center">
	<img
		width="400"
		alt="standard-compliant DataFrame"
		src="https://github.com/MarcoGorelli/impl-dataframe-api/assets/33491632/fb4bc907-2b85-4ad7-8d13-c2b9912b97f5">
</h1>

Work-in-progress POC of what the [DataFrame Standard](https://data-apis.org/dataframe-api/draft/index.html)
might look like for pandas and polars.

Example
-------

Say you want to write a function which selects rows in a DataFrame based on whether the z-score
of a column is between -3 and 3.
In `pandas` you might write:
```python
def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    z_score = (df[column] - df[column].mean())/df[column].std()
    return df[z_score.between(-3, 3)]
```
whereas in polars, you might write:
```python
def remove_outliers(df: pl.DataFrame, column: str) -> pl.DataFrame:
    z_score = ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std())
    return df.filter(z_score.is_between(-3, 3))
```

Using the DataFrame Standard, however, it would be possible to write portable code which would work
for both libraries, allowing library authours to write DataFrame-agnostic code!
```python
def remove_outliers(df, column):
    df_standard = df.__dataframe_standard__()
    col = df_standard.get_column_by_name(column)
    z_score = (col - col.mean()) / col.std()
    return df_standard.get_rows_by_mask((z_score > -3) & (z_score < 3)).dataframe
```
Not only with this work with both pandas and polars, it'll also work with any other DataFrame library
which has an implementation of the DataFrame Standard!

Note: this has not yet been upstreamed into pandas nor polars, so the snippet above does not (yet) work
out-of-the-box.

How to try this out
-------------------

Here's an example of how you can try this out:
```python
import pandas as pd
import pandas_standard  # Necessary to monkey-patch the `__dataframe_standard__` attribute.

df = pd.DataFrame({'a': [1,2,3]})
df_std = df.__dataframe_standard__()
```
The object `df_std` is a Standard-compliant DataFrame.

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
