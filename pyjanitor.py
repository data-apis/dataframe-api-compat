import re
import pandas as pd
import polars as pl
from janitor.functions.clean_names import (
    _change_case,
    _normalize_1,
    _remove_special,
    _strip_accents,
    _strip_underscores_func,
)

from dataframe_standard import dataframe_standard

def clean_names_with_standard(
    df,
    strip_underscores=None,
    case_type='lower',
    remove_special=False,
    strip_accents=True,
    preserve_original_columns=True,
    enforce_string=True,
    truncate_limit=None,
):
    df = dataframe_standard(df)

    if enforce_string:
        # not necessary, as the Standard already
        # imposes string-only names
        pass

    mapping = {}
    for old_label in df.column_names:
        new_label = _change_case(old_label, case_type)
        new_label = _normalize_1(new_label)
        if remove_special:
            new_label = _remove_special(new_label)
        if strip_accents:
            new_label = _strip_accents(new_label)
        new_label = re.sub("_+", "_", new_label)
        new_label = _strip_underscores_func(new_label, strip_underscores)
        new_label = new_label[:truncate_limit]
        mapping[old_label] = new_label
    df = df.rename(mapping)

    # Store the original column names, if enabled by user
    if preserve_original_columns:
        # This writes to df.__dict__ - is that allowed?
        pass
    return df.df

pd_df = pd.DataFrame(
    {
        "Aloha": range(3),
        "Bell Chart": range(3),
        "Animals@#$%^": range(3)
    }
)
pl_df = pl.from_pandas(pd_df)
print(pd_df)
print(clean_names_with_standard(pd_df))
print(clean_names_with_standard(pl_df))
