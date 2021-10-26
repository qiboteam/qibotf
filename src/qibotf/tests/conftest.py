import pytest
from qibotf.custom_operators import TensorflowCustomBackend
K = TensorflowCustomBackend()


@pytest.fixture
def precision(precision_name):
    original_precision = K.precision
    K.set_precision(precision_name)
    yield
    K.set_precision(original_precision)


@pytest.fixture
def threads(num_threads):
    original_threads = K.nthreads
    K.set_threads(num_threads)
    yield
    K.set_threads(original_threads)


def pytest_generate_tests(metafunc):
    if "precision_name" in metafunc.fixturenames:
        metafunc.parametrize("precision_name", ["double", "single"])
    if "num_threads" in metafunc.fixturenames:
        metafunc.parametrize("num_threads", [1, 4])
