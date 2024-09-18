import pytest

from vcub_keeper.transform.features_factory import get_transactions_out, get_transactions_in
from vcub_keeper.reader.reader import read_activity_vcub
from vcub_keeper.config import ROOT_TESTS_DATA


def read_activity_data(file_name="activite_data.csv"):
    """
    Read test csv activity station's data.
    From notebooks/04_tests/03_test_data_activite.ipynb
    """

    return read_activity_vcub(ROOT_TESTS_DATA + file_name)


@pytest.mark.benchmark
def test_benchmark_get_transaction_out(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_out)
    """

    activity_data_feature = get_transactions_out(activite_data)


@pytest.mark.benchmark
def test_benchmark_get_transaction_in(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_in)
    """

    activity_data_feature = get_transactions_in(activite_data)
