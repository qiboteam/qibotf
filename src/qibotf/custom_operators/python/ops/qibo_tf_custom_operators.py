"""Qibo backend that applies the Tensorflow custom operators."""
import os
import tensorflow as tf
import qibo # pylint: disable=E0401
from qibo.backends.abstract import AbstractBackend, AbstractCustomOperators # pylint: disable=E0401
from qibo.backends.tensorflow import TensorflowBackend # pylint: disable=E0401
from qibo.config import raise_error # pylint: disable=E0401


class TensorflowCustomBackend(TensorflowBackend, AbstractCustomOperators):

    description = "Uses precompiled primitives to apply gates to states. " \
                  "This is the fastest simulation engine."

    def __init__(self):
        TensorflowBackend.__init__(self)
        AbstractCustomOperators.__init__(self)
        self.name = "qibotf"
        self.is_custom = True
        if "OMP_NUM_THREADS" in os.environ:  # pragma: no cover
            # environment variable not used in CI
            self.set_threads(int(os.environ.get("OMP_NUM_THREADS")))

        # check if qibotf and tf versions are compatible
        from qibotf import __target_tf_version__
        if tf.__version__ != __target_tf_version__:  # pragma: no cover
            # CI has the proper Tensorflow version
            raise_error(RuntimeError,
                        "qibotf and TensorFlow versions do not match. "
                        "Please check the qibotf documentation.")
        # check if qibotf and qibo versions are compatible
        if qibo.__version__ < "0.1.7":  # pragma: no cover
            # CI has the proper qibo version
            raise_error(RuntimeError,
                        "qibotf requires qibo version higher than 0.1.7. "
                        "Please upgrade qibo or downgrade qibotf.")

        # load custom operators
        from tensorflow.python.framework import load_library  # pylint: disable=no-name-in-module
        from tensorflow.python.platform import resource_loader  # pylint: disable=no-name-in-module
        if tf.config.list_physical_devices("GPU"):  # pragma: no cover
            # case not covered by GitHub workflows because it requires GPU
            library_path = '_qibo_tf_custom_operators_cuda.so'
        else:
            library_path = '_qibo_tf_custom_operators.so'

        self.module = load_library.load_op_library(
                resource_loader.get_path_to_datafile(library_path))

        # enable multi-GPU if no macos
        import sys
        if sys.platform != "darwin":
            self.supports_multigpu = True

        # no gradient support for custom operators
        self.supports_gradients = False

    def set_threads(self, nthreads):
        AbstractBackend.set_threads(self, nthreads)

    def initial_state(self, nqubits, is_matrix=False):
        return self.module.initial_state(nqubits, self.dtypes('DTYPECPX'),
                                         is_matrix=is_matrix,
                                         omp_num_threads=self.nthreads)

    def sample_frequencies(self, probs, nshots):
        from qibo.config import SHOT_METROPOLIS_THRESHOLD # pylint: disable=E0401
        if nshots < SHOT_METROPOLIS_THRESHOLD:
            return super().sample_frequencies(probs, nshots)
        # Generate random seed using tf
        dtype = self.dtypes('DTYPEINT')
        seed = self.backend.random.uniform(
            shape=tuple(), maxval=int(1e8), dtype=dtype)
        nqubits = int(self.np.log2(tuple(probs.shape)[0]))
        shape = self.cast(2 ** nqubits, dtype='DTYPEINT')
        frequencies = self.zeros(shape, dtype='DTYPEINT')
        frequencies = self.module.measure_frequencies(
            frequencies, probs, nshots, nqubits, seed, self.nthreads)
        return frequencies

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None):  # pragma: no cover
        raise_error(NotImplementedError)

    def einsum_call(self, cache, state, matrix):  # pragma: no cover
        raise_error(NotImplementedError)

    def create_gate_cache(self, gate):
        cache = self.GateCache()
        qubits = [gate.nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(gate.nqubits - q - 1 for q in gate.target_qubits)
        cache.qubits_tensor = self.cast(sorted(qubits), "int32")
        if gate.density_matrix:
            cache.target_qubits_dm = [q + gate.nqubits for q in gate.target_qubits]
        return cache

    def _state_vector_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.cache.qubits_tensor, gate.nqubits, gate.target_qubits)

    def state_vector_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.custom_op_matrix, gate.cache.qubits_tensor, # pylint: disable=E1121
                       gate.nqubits, gate.target_qubits)

    def _density_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        state = gate_op(state, gate.cache.qubits_tensor + gate.nqubits, 2 * gate.nqubits, gate.target_qubits)
        state = gate_op(state, gate.cache.qubits_tensor, 2 * gate.nqubits, gate.cache.target_qubits_dm)
        return state

    def density_matrix_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        state = gate_op(state, gate.custom_op_matrix, gate.cache.qubits_tensor + gate.nqubits, # pylint: disable=E1121
                        2 * gate.nqubits, gate.target_qubits)
        adjmatrix = self.conj(gate.custom_op_matrix)
        state = gate_op(state, adjmatrix, gate.cache.qubits_tensor,
                        2 * gate.nqubits, gate.cache.target_qubits_dm)
        return state

    def _density_matrix_half_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.cache.qubits_tensor + gate.nqubits, 2 * gate.nqubits, gate.target_qubits)

    def density_matrix_half_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.custom_op_matrix, gate.cache.qubits_tensor + gate.nqubits, # pylint: disable=E1121
                       2 * gate.nqubits, gate.target_qubits)

    def _result_tensor(self, result):
        n = len(result)
        result = sum(2 ** (n - i - 1) * r for i, r in enumerate(result))
        return self.cast(result, dtype="DTYPEINT")

    def state_vector_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        return self.module.collapse_state(state, gate.cache.qubits_tensor, result,
                                      gate.nqubits, True, self.nthreads)

    def density_matrix_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        qubits = gate.cache.qubits_tensor
        state = self.module.collapse_state(state, qubits + gate.nqubits, result,
                                           2 * gate.nqubits, False, self.nthreads)
        state = self.module.collapse_state(state, qubits, result, 2 * gate.nqubits,
                                           False, self.nthreads)
        return state / self.trace(state)

    def compile(self, func):
        return func

    def transpose_state(self, pieces, state, nqubits, order):
        return self.module.transpose_state(pieces, state, nqubits, order, self.nthreads)

    def apply_gate(self, state, gate, qubits, nqubits, targets):
        """Applies arbitrary one-qubit gate to a state vector.

        Modifies ``state`` in-place.
        Gates can be controlled to multiple qubits.

        Args:
            state (tf.Tensor): State vector of shape ``(2 ** nqubits,)``.
            gate (tf.Tensor): Gate matrix of shape ``(2, 2)``.
            qubits (tf.Tensor): Tensor that contains control and target qubits in
                sorted order. See :meth:`qibo.backends.abstract.TensorflowCustomBackend.cache..qubits_tensor`
                for more details.
            nqubits (int): Total number of qubits in the state vector.
            target (int): Qubit ID that the gate will act on.
                Must be smaller than ``nqubits``.

        Return:
            state (tf.Tensor): State vector of shape ``(2 ** nqubits,)`` after
                ``gate`` is applied.
        """
        return self.module.apply_gate(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_x(self, state, qubits, nqubits, targets):
        return self.module.apply_x(state, qubits, nqubits, *targets, self.nthreads)

    def apply_y(self, state, qubits, nqubits, targets):
        return self.module.apply_y(state, qubits, nqubits, *targets, self.nthreads)

    def apply_z(self, state, qubits, nqubits, targets):
        return self.module.apply_z(state, qubits, nqubits, *targets, self.nthreads)

    def apply_z_pow(self, state, gate, qubits, nqubits, targets):
        return self.module.apply_z_pow(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_two_qubit_gate(self, state, gate, qubits, nqubits, targets):
        return self.module.apply_two_qubit_gate(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_swap(self, state, qubits, nqubits, targets):
        return self.module.apply_swap(state, qubits, nqubits, *targets, self.nthreads)

    def apply_fsim(self, state, gate, qubits, nqubits, targets):
        return self.module.apply_fsim(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_multi_qubit_gate(self, state, gate, qubits, nqubits, targets):
        n = len(targets)
        raise_error(NotImplementedError,
                    "qibotf supports up to two-qubit gates but {} "
                    "targets were given. Please switch to another "
                    "backend to execute this operation.".format(n))

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        return self.module.collapse_state(state, qubits, result, nqubits, normalize, self.nthreads)

    def swap_pieces(self, piece0, piece1, new_global, nlocal):
        with self.on_cpu():
            return self.module.swap_pieces(piece0, piece1, new_global, nlocal, self.nthreads)
