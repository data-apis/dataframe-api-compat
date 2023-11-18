# ruff: noqa
import sys

sys.path.append("..")

import pandas as pd

members = [
    i
    for i in pd.Series([1], name="a").__column_consortium_standard__().__dir__()
    if not i.startswith("_")
]

with open("pandas-column.md") as fd:
    content = fd.read()

members_txt = "\n      - ".join(sorted(members))

start = content.index("members:")
end = content.index("show_signature")
content = content[:start] + f"members:\n      - {members_txt}" + content[end:]

with open("pandas-column.md", "w") as fd:
    fd.write(content)
