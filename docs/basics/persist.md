# Persist

If you're used to pandas, then you might have been surprised to see `DataFrame.persist` is
an example from [column.md](column.md). But...what is it?

The basic idea is:

> If you call `.persist`, then computation prior to this point won't be repeated.

If this is confusing, don't worry, we'll see an example. If you follow the
rule:

**Call `.persist` as little and as late as possible, ideally just once per function / dataframe**

then you'll likely be fine.

## Why do we need it?

The `dataframe-api-compat` package is written with lazy computation in mind. For the Polars implementation,
all objects are backed by lazy constructs:

- `DataFrame`:
  - by default, backed by `polars.LazyFrame`
  - if you call `persist`, backed by `polars.DataFrame`
- `Column`:
  - by default, backed by `polars.Expr`
  - if you call `persist`, or if you called `persist` on
    the dataframe it was derived from, backed by `polars.Series`
- `Scalar`:
  - by default, backed by `polars.Expr`
  - if you call `persist`, or if you called `persist` on
    the dataframe or column it was derived from, backed by
    a Python scalar.

All operations can be done lazily, except for:
- `DataFrame.to_array()`,
- `Column.to_array()`,
- `DataFrame.shape`,
- bringing a `Scalar` into Python, e.g. `float(df.col('a').mean())`

Let's see what you need to do when using `dataframe-api-compat` to achieve the above.

## Example 1: splitting a dataframe and converting to array

Say you have a DataFrame `df`, and want to split it into `features` and `target`, and want
to convert both to numpy arrays. Let's see how you can achieve this.

If you try running the code below

```python exec="1" source="above" session="persist-ex1"
def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    x_train = df_s.drop_columns('y').to_array()
    y_train = df_s.col('y').to_array()
    return x_train, y_train
```

you'll get an error like:
```python
Method requires you to call `.persist` first.
```

Here's how to fix up the function so it runs: we add a single `persist`,
just once, before splitting the dataframe within numpy:

```python exec="1" source="above" session="persist-ex1"
import numpy as np

def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    arr = df_s.persist().to_array()
    target_idx = df_s.column_names.index('y')
    x_train = np.delete(arr, target_idx, axis=1)
    y_train = arr[:, target_idx]
    return x_train, y_train
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="persist-ex1"
    import pandas as pd

    df = pd.DataFrame({'x': [-1, 1, 3], 'y': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="persist-ex1"
    import polars as pl

    df = pl.DataFrame({'x': [-1, 1, 3], 'y': [3, 5, -3]})
    print(my_func(df))
    ```

If you find yourself repeatedly calling `persist`, you might be re-triggering
the same computation multiple times.
