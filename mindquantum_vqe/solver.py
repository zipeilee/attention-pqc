from __future__ import annotations

"""Numerical VQE solvers shared by the direct and MindSpore-backed paths."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import math
import time

import numpy as np

from .circuits import build_circuit
from .config import OptimizerConfig, SweepConfig, VQEConfig, collect_environment_info
from .hamiltonians import build_hamiltonian, hamiltonian_metadata, hamiltonian_terms, term_to_mindquantum_string


@dataclass
class PointResult:
    """Optimization result for a single control-parameter value."""

    lambda_value: float
    status: str
    converged: bool
    best_energy: float
    final_energy: float
    n_iters: int
    best_params: List[float]
    param_names: List[str]
    history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    init_source: str = "unknown"
    reference_energy: Optional[float] = None
    energy_gap: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into JSON-serializable data."""
        payload = asdict(self)
        payload["best_params"] = list(self.best_params)
        payload["param_names"] = list(self.param_names)
        return payload


@dataclass
class SweepResult:
    """Collection of point-wise VQE results over a parameter sweep."""

    points: List[PointResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sweep metadata and all contained point results."""
        return {
            "metadata": self.metadata,
            "points": [point.to_dict() for point in self.points],
        }


class AdamState:
    """Minimal Adam optimizer state used by the NumPy-based solver path."""

    def __init__(self, size: int, beta1: float, beta2: float, epsilon: float) -> None:
        self.m = np.zeros(size, dtype=float)
        self.v = np.zeros(size, dtype=float)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step = 0

    def update(self, params: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
        """Apply one Adam update step and return the new parameter vector."""
        self.step += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.beta1 ** self.step)
        v_hat = self.v / (1.0 - self.beta2 ** self.step)
        return params - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class ExpectationWithGrad:
    """Wrap MindQuantum expectation/gradient evaluation behind a NumPy API."""

    def __init__(self, system_n_qubits: int, terms: List[Any], circuit: Any, backend: str) -> None:
        try:
            from mindquantum.simulator import Simulator
        except ImportError as exc:
            raise ImportError(
                "MindQuantum is required to evaluate expectation values and gradients."
            ) from exc
        self.simulator = Simulator(backend, system_n_qubits)
        self.terms = terms
        self.circuit = circuit
        self.backend = backend

        try:
            hamiltonians = []
            from mindquantum.core.operators import Hamiltonian, QubitOperator

            for coefficient, term in terms:
                hamiltonians.append(Hamiltonian(QubitOperator(term_to_mindquantum_string(term), coefficient)))
            self.grad_ops = self.simulator.get_expectation_with_grad(hamiltonians, circuit)
            self._batched = True
        except Exception:
            hamiltonian = build_hamiltonian_from_terms(terms)
            self.grad_ops = self.simulator.get_expectation_with_grad(hamiltonian, circuit)
            self._batched = False

    def __call__(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate the current energy and gradient at the given parameters."""
        params = np.asarray(params, dtype=float)
        result = self.grad_ops(params)
        if self._batched:
            energy = float(np.real(np.asarray(result[0]).reshape(-1).sum()))
            grad = np.real(np.asarray(result[1])).reshape(-1)
        else:
            energy = float(np.real(np.asarray(result[0]).reshape(-1)[0]))
            grad = np.real(np.asarray(result[1])).reshape(-1)
        return energy, grad


def build_hamiltonian_from_terms(terms: List[Any]) -> Any:
    """Rebuild a single Hamiltonian object from internal weighted terms."""
    try:
        from mindquantum.core.operators import Hamiltonian, QubitOperator
    except ImportError as exc:
        raise ImportError(
            "MindQuantum is required to construct Hamiltonians."
        ) from exc
    operator = QubitOperator()
    for coefficient, term in terms:
        operator += QubitOperator(term_to_mindquantum_string(term), coefficient)
    return Hamiltonian(operator)


def sweep_values(config: SweepConfig) -> List[float]:
    """Expand the sweep configuration into an ordered list of control values."""
    if config.lambda_values:
        return [float(value) for value in config.lambda_values]
    if not config.enabled:
        return [float(config.single_lambda)]
    values: List[float] = []
    current = config.lambda_start
    while current <= config.lambda_stop + config.lambda_step * 1e-9:
        values.append(round(float(current), 12))
        current += config.lambda_step
    return values


def initialize_params(n_params: int, optimizer: OptimizerConfig) -> np.ndarray:
    """Create an initial parameter vector using the configured strategy."""
    rng = np.random.default_rng(optimizer.seed)
    if optimizer.init_strategy == "zeros":
        return np.zeros(n_params, dtype=float)
    if optimizer.init_strategy == "ones":
        return np.ones(n_params, dtype=float)
    if optimizer.init_strategy == "random":
        return rng.normal(0.0, optimizer.init_scale, size=n_params).astype(float)
    raise ValueError(f"Unsupported init strategy: {optimizer.init_strategy}")


def build_expectation_with_grad(config: VQEConfig, control_value: float) -> tuple[Any, Any, Any]:
    """Construct the circuit, Hamiltonian, and evaluator for one sweep point."""
    circuit_info = build_circuit(config.system)
    hamiltonian, terms = build_hamiltonian(config.system, control_value)
    evaluator = ExpectationWithGrad(
        system_n_qubits=config.system.n_qubits,
        terms=terms,
        circuit=circuit_info.circuit,
        backend=config.runtime.backend,
    )
    return circuit_info, hamiltonian, evaluator


def exact_reference_energy(config: VQEConfig, control_value: float) -> Optional[float]:
    """Diagonalize small systems exactly to provide a reference baseline."""
    if config.system.n_qubits > config.runtime.exact_reference_max_qubits:
        return None
    terms = hamiltonian_terms(config.system, control_value)
    dim = 2 ** config.system.n_qubits
    pauli = {
        "I": np.array([[1, 0], [0, 1]], dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    ham = np.zeros((dim, dim), dtype=complex)
    for coefficient, term in terms:
        mats: List[np.ndarray] = []
        term_map = {idx: gate for idx, gate in term}
        for qubit in range(config.system.n_qubits):
            mats.append(pauli[term_map.get(qubit, "I")])
        term_matrix = mats[0]
        for mat in mats[1:]:
            term_matrix = np.kron(term_matrix, mat)
        ham += coefficient * term_matrix
    eigenvalues = np.linalg.eigvalsh(ham)
    return float(np.real(eigenvalues.min()))


def solve_single_point(
    config: VQEConfig,
    control_value: float,
    initial_params: Optional[np.ndarray] = None,
    init_source: str = "fresh",
) -> PointResult:
    """Optimize one Hamiltonian instance with the NumPy + MindQuantum path."""
    config.validate()
    circuit_info, _, evaluator = build_expectation_with_grad(config, control_value)
    params = (
        np.asarray(initial_params, dtype=float).copy()
        if initial_params is not None
        else initialize_params(circuit_info.n_params, config.optimizer)
    )
    if params.shape[0] != circuit_info.n_params:
        raise ValueError("Initial parameter size does not match circuit parameter count.")

    optimizer_state = AdamState(
        size=circuit_info.n_params,
        beta1=config.optimizer.beta1,
        beta2=config.optimizer.beta2,
        epsilon=config.optimizer.epsilon,
    )

    best_params = params.copy()
    best_energy = math.inf
    history: List[float] = []
    grad_norm_history: List[float] = []
    previous_energy = math.inf
    patience_count = 0
    status = "max_iters_reached"
    start_time = time.time()

    for step in range(1, config.optimizer.max_iters + 1):
        energy, grad = evaluator(params)
        grad_norm = float(np.linalg.norm(grad))
        history.append(float(energy))
        grad_norm_history.append(grad_norm)

        if energy < best_energy:
            best_energy = float(energy)
            best_params = params.copy()

        if abs(energy - previous_energy) < config.optimizer.tol:
            patience_count += 1
        else:
            patience_count = 0
        previous_energy = float(energy)

        if patience_count >= config.optimizer.patience:
            status = "converged"
            break

        params = optimizer_state.update(params, grad, config.optimizer.learning_rate)

    final_energy, _ = evaluator(params)
    reference_energy = exact_reference_energy(config, control_value)
    energy_gap = None if reference_energy is None else float(abs(reference_energy - best_energy))
    converged = status == "converged"

    return PointResult(
        lambda_value=float(control_value),
        status=status,
        converged=converged,
        best_energy=float(best_energy),
        final_energy=float(final_energy),
        n_iters=len(history),
        best_params=best_params.astype(float).tolist(),
        param_names=list(circuit_info.param_names),
        history=history,
        grad_norm_history=grad_norm_history,
        init_source=init_source,
        reference_energy=reference_energy,
        energy_gap=energy_gap,
        metadata={
            "ansatz": circuit_info.ansatz,
            "n_qubits": config.system.n_qubits,
            "depth": config.system.depth,
            "hamiltonian": config.system.hamiltonian,
            "backend": config.runtime.backend,
            "elapsed_seconds": time.time() - start_time,
            **hamiltonian_metadata(config.system, control_value),
        },
    )


def solve_sweep(config: VQEConfig) -> SweepResult:
    """Run the configured parameter sweep, optionally reusing warm starts."""
    config.validate()
    values = sweep_values(config.sweep)
    points: List[PointResult] = []
    next_initial: Optional[np.ndarray] = None

    for index, control_value in enumerate(values):
        if config.sweep.warm_start and next_initial is not None:
            current_initial = next_initial.copy()
            init_source = "warm_start"
        else:
            current_initial = None
            init_source = "fresh"
        point = solve_single_point(
            config,
            control_value=control_value,
            initial_params=current_initial,
            init_source=init_source,
        )
        points.append(point)
        if config.sweep.warm_start:
            next_initial = np.asarray(point.best_params, dtype=float)

    return SweepResult(
        points=points,
        metadata={
            "environment": collect_environment_info(),
            "mode": config.runtime.mode,
            "warm_start": config.sweep.warm_start,
            "n_points": len(points),
            "lambda_values": values,
        },
    )
