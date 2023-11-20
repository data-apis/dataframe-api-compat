# Scalar

In [column.md](column.md), you learned how to write a dataframe-agnostic function
involving both dataframes and columns.

But what if we want to extract scalars as well?

## Example 1: center features

Let's try writing a function which, for each column, subtracts its mean.

```python exec="1" source="above" session="ex1"
def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    new_columns = [col - col.mean() for col in df_s.columns_iter()]
    df_s = df_s.assign(*new_columns)
    return df_s.dataframe
```

Let's run it:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df).collect())
    ```

The output looks as expected. `df.col(column_name).mean()` returns a `Scalar`, which
can be combined with a `Column` from the same dataframe. Just [like we saw for `Column`s](column.md),
scalars from different dataframes cannot be compared - you'll first need to join the underlying
dataframes.

## Example 2: Store mean of each column as Python float

We saw in the above example that `df.col(column_name).mean()` returns a `Scalar`, which may
be lazy. In particular, it's not a Python scalar. So, how would we force execution and store
a Python scalar, in a dataframe-agnostic manner?

```python exec="1" source="above" session="ex1"
def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    # We'll learn more about `persist` in the next page
    df_s = df_s.mean().persist()
    means = []
    for column_name in df_s.column_names:
        mean = float(df_s.col(column_name).get_value(0))
        means.append(mean)
    return means
```

Let's run it:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

We'll learn more about `DataFrame.persist` in the next slide.
