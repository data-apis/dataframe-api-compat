Installation
------------
```
python3.10 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-dev.txt
```
Testing
-------
```
pytest test_standard.py --cov=dataframe_api_compat --cov=test_standard --cov-fail-under=100
```
100% branch coverage isn't the objective - it's the bare minimum.

Linting
-------
```
pre-commit run --all-files
```

Type Checking
-------------

First, clone the [dataframe_standard](https://github.com/data-apis/dataframe-api) to some
local path. Then, run:
```console
MYPYPATH=<path to dataframe-api/spec/API_specification> mypy dataframe_api_compat
MYPYPATH=<path to dataframe-api/spec/API_specification> mypy polars_standard.py
```

For example, if you cloned both repos in the same place, this could be:
```console
MYPYPATH=../dataframe-api/spec/API_specification/ mypy dataframe_api_compat
```
