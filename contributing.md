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
pytest tests --cov=dataframe_api_compat/pandas_standard --cov-fail-under=100 --library pandas-numpy
pytest tests --cov=dataframe_api_compat/pandas_standard --cov-fail-under=100 --library pandas-nullable
pytest tests --cov=dataframe_api_compat/polars_standard --cov-fail-under=100 --library polars-lazy
# prerequisite: `pip install -r requirements-dev-modin.txt`
pytest tests --cov=dataframe_api_compat/modin_standard --cov-fail-under=100 --library modin
```
100% branch coverage isn't the objective - it's the bare minimum.

**Note**: there is also requirement about 100% coverage in `tests` folder, however, to achieve one
need to combine coverage from different implementations. How this is done can be seen in the `tox.yml` CI flow.

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
MYPYPATH=<path to dataframe-api/spec/API_specification> mypy dataframe_api_compat/pandas_standard dataframe_api_compat/polars_standard
# prerequisite: `pip install -r requirements-dev-modin.txt`
MYPYPATH=<path to dataframe-api/spec/API_specification> mypy dataframe_api_compat/modin_standard
```

For example, if you cloned both repos in the same place, this could be:
```console
MYPYPATH=../dataframe-api/spec/API_specification/ mypy dataframe_api_compat/pandas_standard dataframe_api_compat/polars_standard
MYPYPATH=../dataframe-api/spec/API_specification/ mypy dataframe_api_compat/modin_standard
```
