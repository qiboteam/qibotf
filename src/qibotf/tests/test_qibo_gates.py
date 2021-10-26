"""Test qibotf backend when used to apply qibo gates."""
import pytest
import numpy as np
import qibo
from qibo import K, gates
from qibo.models import Circuit


def random_complex(shape):
    return np.random.random(shape) + 1j * np.random.random(shape)


@pytest.mark.parametrize("nqubits,gate",
                         [(1, "H"), (1, "X"), (1, "Y"), (1, "Z"),
                          (2, "CNOT"), (2, "CZ")])
def test_circuit_execution(nqubits, gate, density_matrix):
    shape = (1 + density_matrix) * (2 ** nqubits,)
    initial_state = random_complex(shape)

    qibo.set_backend("qibotf")
    c = Circuit(nqubits, density_matrix=density_matrix)
    c.add(getattr(gates, gate)(*range(nqubits)))
    final_state = c(np.copy(initial_state)).state()

    qibo.set_backend("numpy")
    c = Circuit(nqubits, density_matrix=density_matrix)
    c.add(getattr(gates, gate)(*range(nqubits)))
    target_state = c(np.copy(initial_state)).state()
    K.assert_allclose(final_state, target_state)
