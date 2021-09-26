import pytest
import numpy as np
import qibo
from qibotf import custom_operators as op


_atol = 1e-6

def qubits_tensor(nqubits, targets, controls=[]):
    qubits = list(nqubits - np.array(controls) - 1)
    qubits.extend(nqubits - np.array(targets) - 1)
    qubits = sorted(qubits)
    return qubits


def random_complex(shape, dtype="complex128"):
    return (np.random.random(shape) + 1j * np.random.random(shape)).astype(dtype)


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(3, [0, 1, 2], []), (4, [2, 1, 3], []),
                          (5, [0, 2, 3], []), (8, [2, 6, 3], []),
                          (5, [0, 2, 3, 4], []),
                          (8, [0, 4, 2, 5, 7], []),
                          (4, [2, 1, 3], [0]), (5, [0, 2, 3], [1]),
                          (8, [2, 6, 3], [4, 7]), (5, [0, 2, 3, 4], [1]),
                          (8, [0, 4, 2, 5, 7], [1, 3]),
                          (10, [0, 4, 2, 5, 9], [1, 3, 7, 8])
                          ])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("threads", [1, 4])
def test_apply_multiqubit_gate(nqubits, targets, controls, dtype, threads):
    qibo.set_backend("numpy")
    state = random_complex((2 ** nqubits,), dtype=dtype)
    state = state / np.sqrt(np.sum(np.abs(state) ** 2))
    rank = 2 ** len(targets)
    matrix = random_complex((rank, rank), dtype=dtype)

    gate = qibo.gates.Unitary(matrix, *targets).controlled_by(*controls)
    target_state = gate(np.copy(state))

    qubits = qubits_tensor(nqubits, targets, controls)
    state = op.apply_multi_qubit_gate(state, matrix, qubits, targets, nqubits, threads)
    np.testing.assert_allclose(state, target_state, atol=_atol)
