# Quick start

## Prerequisites

Please start by following the [installation instructions](installation.md)

Then, please install the following:

- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [Polars](https://pola-rs.github.io/polars/user-guide/installation/)

## Simple example

Create a Python file `t.py` with the following content:

```python
import pandas as pd
import polars as pl


def my_function(df_any):
    df = df_any.__dataframe_consortium_standard__(api_version='2023.11-beta')
    column_names = df.column_names
    return column_names


df_pandas = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df_polars = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

print('pandas result: ', my_function(df_pandas))
print('Polars result: ', my_function(df_polars))
```

If you run `python t.py` and your output looks like this:
```
pandas result: ['a', 'b']
Polars result: ['a', 'b']
```

then all your installations worked perfectly.

Let's learn about what you just did, and what `dataframe-api-compat` can do for you.
