"""Test Tensorflow custom operators."""
import pytest
import itertools
import numpy as np
import tensorflow as tf
import qibo
from qibo import K
qibo.set_backend("qibotf")


def qubits_tensor(nqubits, targets, controls=[]):
    qubits = list(nqubits - np.array(controls) - 1)
    qubits.extend(nqubits - np.array(targets) - 1)
    qubits = sorted(qubits)
    return qubits


def random_complex(shape, dtype="complex128"):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    return tf.cast(x, dtype=dtype)


@pytest.mark.parametrize("compile", [False, True])
def test_initial_state(compile, nthreads):
    """Check that initial_state updates first element properly."""
    def apply_operator():
        """Apply the initial_state operator"""
        return K.initial_state(nqubits=4, is_matrix=False)

    func = apply_operator
    if compile:
        func = K.compile(apply_operator)
    final_state = func()
    exact_state = np.array([1] + [0]*15)
    K.assert_allclose(final_state, exact_state)


@pytest.mark.parametrize("nqubits,targets,results",
                         [(2, [0], [1]), (2, [1], [0]), (3, [1], [1]),
                          (4, [1, 3], [1, 0]), (5, [1, 2, 4], [0, 1, 1]),
                          (15, [4, 7], [0, 0]), (16, [8, 12, 15], [1, 0, 1])])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_collapse_state(nqubits, targets, results, dtype, nthreads):
    """Check ``collapse_state`` kernel."""
    atol = 1e-7 if dtype == np.complex64 else 1e-14
    state = random_complex((2 ** nqubits,), dtype=dtype)
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = r
    slicer = tuple(slicer)
    initial_state = np.reshape(state, nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)

    qubits = sorted(nqubits - np.array(targets) - 1)
    b2d = 2 ** np.arange(len(results) - 1, -1, -1)
    result = np.array(results).dot(b2d)
    state = K.collapse_state(state, qubits, result, nqubits, True)
    K.assert_allclose(state, target_state, atol=atol)


def check_unimplemented_error(func, *args):  # pragma: no cover
    # method not tested by GitHub workflows because it requires GPU
    from tensorflow.python.framework import errors_impl  # pylint: disable=no-name-in-module
    error = errors_impl.UnimplementedError
    with pytest.raises(error):
        func(*args)


@pytest.mark.parametrize("nqubits", [3, 4, 7, 8, 9, 10])
@pytest.mark.parametrize("ndevices", [2, 4, 8])
def test_transpose_state(nqubits, ndevices, nthreads):
    for _ in range(10):
        # Generate global qubits randomly
        all_qubits = np.arange(nqubits)
        np.random.shuffle(all_qubits)
        qubit_order = list(all_qubits)
        state = random_complex((2 ** nqubits,))

        state_tensor = state.numpy().reshape(nqubits * (2,))
        target_state = np.transpose(state_tensor, qubit_order).ravel()

        new_state = tf.zeros_like(state)
        shape = (ndevices, int(state.shape[0]) // ndevices)
        state = tf.reshape(state, shape)
        pieces = [state[i] for i in range(ndevices)]
        if "GPU" in K.default_device:  # pragma: no cover
            # case not tested by GitHub workflows because it requires GPU
            check_unimplemented_error(K.transpose_state,
                                      pieces, new_state, nqubits, qubit_order)
        else:
            new_state = K.transpose_state(
                pieces, new_state, nqubits, qubit_order)
            K.assert_allclose(new_state, target_state)


@pytest.mark.parametrize("nqubits", [4, 5, 7, 8, 9, 10])
def test_swap_pieces_zero_global(nqubits, nthreads):
    if "GPU" in K.default_device:  # pragma: no cover
        pytest.skip("Skipping ``swap_pieces`` test as it is not implemented  for GPU.")
    state = random_complex((2 ** nqubits,))
    target_state = tf.cast(np.copy(state), dtype=np.complex128)
    shape = (2, int(state.shape[0]) // 2)
    state = tf.reshape(state, shape)

    for _ in range(10):
        local = np.random.randint(1, nqubits)

        qubits_t = qubits_tensor(nqubits, [0, local])
        target_state = K.apply_swap(target_state, qubits_t, nqubits, (0, local))
        target_state = tf.reshape(target_state, shape)

        piece0, piece1 = state[0], state[1]
        K.swap_pieces(piece0, piece1, local - 1, nqubits - 1)
        K.assert_allclose(piece0, target_state[0])
        K.assert_allclose(piece1, target_state[1])


@pytest.mark.parametrize("nqubits", [5, 7, 8, 9, 10])
def test_swap_pieces(nqubits, nthreads):
    if "GPU" in K.default_device:  # pragma: no cover
        pytest.skip("Skipping ``swap_pieces`` test as it is not implemented for GPU.")
    state = random_complex((2 ** nqubits,))
    target_state = tf.cast(np.copy(state), dtype=state.dtype)
    shape = (2, int(state.shape[0]) // 2)

    for _ in range(10):
        global_qubit = np.random.randint(0, nqubits)
        local_qubit = np.random.randint(0, nqubits)
        while local_qubit == global_qubit:
            local_qubit = np.random.randint(0, nqubits)

        transpose_order = ([global_qubit] + list(range(global_qubit)) +
                           list(range(global_qubit + 1, nqubits)))

        qubits_t = qubits_tensor(nqubits, [global_qubit, local_qubit])
        target_state = K.apply_swap(
            target_state, qubits_t, nqubits, (global_qubit, local_qubit))
        target_state = tf.reshape(target_state, nqubits * (2,))
        target_state = tf.transpose(target_state, transpose_order)
        target_state = tf.reshape(target_state, shape)

        state = tf.reshape(state, nqubits * (2,))
        state = tf.transpose(state, transpose_order)
        state = tf.reshape(state, shape)
        piece0, piece1 = state[0], state[1]
        K.swap_pieces(piece0, piece1, local_qubit - int(global_qubit < local_qubit), nqubits - 1)
        K.assert_allclose(piece0, target_state[0])
        K.assert_allclose(piece1, target_state[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inttype", [np.int32, np.int64])
def test_measure_frequencies(dtype, inttype):
    import sys
    probs = np.ones(16, dtype=dtype) / 16
    frequencies = np.zeros(16, dtype=inttype)
    frequencies = K.module.measure_frequencies(frequencies, probs, nshots=1000,
                                               nqubits=4, omp_num_threads=1,
                                               seed=1234)
    if sys.platform == "linux":
        target_frequencies = [60, 50, 68, 64, 53, 53, 67, 54, 64, 53, 67,
                              69, 76, 57, 64, 81]
    elif sys.platform == "darwin":  # pragma: no cover
        target_frequencies = [57, 51, 62, 63, 55, 70, 52, 47, 75, 58, 63,
                              73, 68, 72, 60, 74]
    assert np.sum(frequencies) == 1000
    K.assert_allclose(frequencies, target_frequencies)


NONZERO = list(itertools.combinations(range(8), r=1))
NONZERO.extend(itertools.combinations(range(8), r=2))
NONZERO.extend(itertools.combinations(range(8), r=3))
NONZERO.extend(itertools.combinations(range(8), r=4))
@pytest.mark.parametrize("nonzero", NONZERO)
def test_measure_frequencies_sparse_probabilities(nonzero):
    import sys
    probs = np.zeros(8, dtype=np.float64)
    for i in nonzero:
        probs[i] = 1
    probs = probs / np.sum(probs)
    frequencies = np.zeros(8, dtype=np.int64)
    frequencies = K.module.measure_frequencies(frequencies, probs, nshots=1000,
                                               nqubits=3, omp_num_threads=1,
                                               seed=1234)
    assert np.sum(frequencies) == 1000
    for i, freq in enumerate(frequencies):
        if i in nonzero:
            assert freq != 0
        else:
            assert freq == 0
