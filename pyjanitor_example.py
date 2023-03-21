# Example of how to use the Standard to implement on of
# pyjanitor's simplest functions (clean_names).

import re
import pandas as pd

from pandas_standard import PandasDataFrame

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
    from janitor.functions.clean_names import (
        _change_case,
        _normalize_1,
        _remove_special,
        _strip_accents,
        _strip_underscores_func,
    )

    df = PandasDataFrame(df)

    if enforce_string:
        # not necessary, as the Standard already
        # imposes string-only names
        pass

    mapping = {}
    for old_label in df.get_column_names():
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
    df = df.rename_columns(mapping)

    # Store the original column names, if enabled by user
    if preserve_original_columns:
        # This writes to df.__dict__ - is that allowed?
        pass
    return df.dataframe

pd_df = pd.DataFrame(
    {
        "Aloha": range(3),
        "Bell Chart": range(3),
        "Animals@#$%^": range(3)
    }
)
print(pd_df)
print(clean_names_with_standard(pd_df))
