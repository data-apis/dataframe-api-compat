import polars as pl
from datetime import date

df = pl.DataFrame(
    {
        "date": pl.date_range(date(2020, 10, 30), date(2021, 6, 1), eager=True),
    }
)
df = df.with_columns(values=pl.arange(0, len(df)))

print(df.groupby_dynamic("date", every="3mo").agg(pl.col("values").sum()))
