# Column

In [dataframe.md](dataframe.md), you learned how to write a dataframe-agnostic function.

We only used DataFrame methods there - but what if we need to operate on its columns?

## Extracting a column


## Example 1: filter based on a column's values

```python exec="1" source="above" session="ex1"
def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.filter(df_s.col('a') > 0)
    return df_s.dataframe
```

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


## Example 2: multiply a column's values by a constant

Let's write a dataframe-agnostic function which multiplies the values in column
`'a'` by 2.

```python exec="1" source="above" session="ex2"
def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.assign(df_s.col('a')*2)
    return df_s.dataframe
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df).collect())
    ```

Note that column `'a'` was overwritten. If we had wanted to add a new column called `'c'` containing column `'a'`'s
values multiplied by 2, we could have used `Column.rename`:

```python exec="1" source="above" session="ex2.1"
def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.assign((df_s.col('a')*2).rename('c'))
    return df_s.dataframe
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex2.1"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex2.1"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df).collect())
    ```

## Example 3: cross-dataframe column comparisons

You might expect a function like the following to just work:

```python exec="1" source="above" session="ex3"
def my_func(df1, df2):
    df1_s = df1.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df2_s = df2.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df1_s.filter(df2_s.col('a') > 0)
    return df_s.dataframe
```

However, if you tried passing two different dataframes to this function, you'd get
a message saying something like:
```python
cannot compare columns from different dataframes
```

This is because `Column`s for the Polars implementation are backed by `polars.Expr`s.
The error is there to ensure that the Polars and pandas implementations behave in the same way.
If you wish to compare columns from different dataframes, you should first join the dataframes.
For example:
```python exec="1" source="above" session="ex3.1"
def my_func(df1, df2):
    df1_s = df1.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df2_s = df2.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df1_s = df1_s.join(
        df2_s.rename_columns({'a': 'a_right'}),
        left_on='b',
        right_on='b',
        how='inner',
    )
    df1_s.filter(df1_s.col('a_right') > 0)
    return df1_s.dataframe
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex3.1"
    import pandas as pd

    df1 = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    df2 = pd.DataFrame({'a': [5, 4], 'b': [5, -3]})
    print(my_func(df1, df2))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex3.1"
    import polars as pl

    df1 = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    df2 = pl.DataFrame({'a': [5, 4], 'b': [5, -3]})
    print(my_func(df1, df2).collect())
    ```
