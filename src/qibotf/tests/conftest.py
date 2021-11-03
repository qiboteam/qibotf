import pytest
import qibo
from qibo import K
qibo.set_backend("qibotf")


@pytest.fixture
def nthreads(num_threads):
    original_threads = K.nthreads
    K.set_threads(num_threads)
    yield
    K.set_threads(original_threads)


def pytest_generate_tests(metafunc):
    if "num_threads" in metafunc.fixturenames:
        metafunc.parametrize("num_threads", [1, 4])
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", ["complex128", "complex64"])
