#PYTHONPATH=../dataframe-api/spec/API_specification/ python check_completeness.py
#. type-check.sh
pytest tests --cov=dataframe_api_compat --cov=tests --cov-fail-under=100 -W error
