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
    ```python exec="true" source="tabbed-left" result="bash" session="ex1"
    import pandas as pd

    df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df_pd))
    ```

=== "Polars"
    ```python exec="true" source="tabbed-left" result="bash" session="ex1"
    import polars as pl

    df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df_pl).collect())
    ```


## Example 2: multiply a column's values by a constant

Let's write a dataframe-agnostic function which multiplies the values in column
`'a'` by 2.

```python
import pandas as pd
import polars as pl

def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.assign(df_s.col('a')*2)
    return df_s.dataframe

df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
print('pandas output:')
print(my_func(df_pd))
print()
print('Polars output:')
print(my_func(df_pl).collect())
```

Note that column `'a'` was overwritten. If we had wanted to add a new column called `'c'` containing column `'a'`'s
values multiplied by 2, we could have used `Column.rename`:
```python
import pandas as pd
import polars as pl

def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.assign((df_s.col('a')*2).rename('c'))
    return df_s.dataframe

df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
print('pandas output:')
print(my_func(df_pd))
print()
print('Polars output:')
print(my_func(df_pl).collect())
```
This outputs:
```
pandas output:
   a  b  c
0 -1  3 -2
1  1  5  2
2  3 -3  6

Polars output:
shape: (3, 3)
┌─────┬─────┬─────┐
│ a   ┆ b   ┆ c   │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ -1  ┆ 3   ┆ -2  │
│ 1   ┆ 5   ┆ 2   │
│ 3   ┆ -3  ┆ 6   │
└─────┴─────┴─────┘
```

## Example 3: cross-dataframe column comparisons

You might expect a function like the following to just work:
```python
def my_func(df1, df2):
    df1_s = df1.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df2_s = df2.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df1_s.filter(df2_s.col('a') > 0)
    return df_s.dataframe

df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
print('pandas output:')
print(my_func(df_pd, df_pd+1))
print()
print('Polars output:')
print(my_func(df_pl, df_pl+1).collect())
```
If you try running this, you'll see:
```
ValueError: cannot compare columns from different dataframes
```
This is because `Column`s for the Polars implementation are backed by `polars.Expr`s.
The error is there to ensure that the Polars and pandas implementations behave in the same way.
If you wish to compare columns from different dataframes, you should first join the dataframes.
For example:
```python exec="on"
import pandas as pd
import polars as pl


def my_func(df1, df2):
    df1_s = df1.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df2_s = df2.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df1_s = df1_s.join(df2_s, left_on=['a', 'b'], right_on=['a', 'b'], how='left')
    df1_s = df1_s.filter(df1_s.col('a') > 0)
    return df1_s.dataframe

df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
print('pandas output:')
print(my_func(df_pd, df_pd+1))
print()
print('Polars output:')
print(my_func(df_pl, df_pl+1).collect())
```
which outputs
```
pandas output:
   a  b
0  1  5
1  3 -3

Polars output:
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 5   │
│ 3   ┆ -3  │
└─────┴─────┘
```
