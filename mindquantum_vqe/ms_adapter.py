from __future__ import annotations

"""MindSpore integration layer for training MindQuantum VQE ansatz parameters."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import time

import numpy as np

from .circuits import CircuitBuildResult, build_circuit
from .config import VQEConfig
from .hamiltonians import build_hamiltonian, hamiltonian_metadata
from .solver import PointResult, SweepResult, exact_reference_energy, initialize_params, sweep_values


@dataclass
class MindSporeVQEArtifacts:
    """Objects needed to drive a MindSpore-backed VQE training loop."""

    grad_ops: Any
    layer: Any
    objective: Any
    optimizer: Any
    train_cell: Any
    circuit_info: CircuitBuildResult
    supports_return_grad: bool
    runtime_context: Dict[str, str]


@dataclass
class MindSporeTrainStepResult:
    """Scalar outputs captured after one MindSpore training step."""

    loss: float
    grad_norm: Optional[float]


@dataclass
class MindSporeLayerSnapshot:
    """Instantaneous layer state used for logging and best-checkpoint tracking."""

    energy: float
    weights: np.ndarray


def initialize_mindspore(runtime_config: Any) -> Dict[str, str]:
    """Initialize the MindSpore execution context from runtime settings."""
    try:
        import mindspore as ms
    except ImportError as exc:
        raise ImportError(
            "MindSpore is required for the MindSpore-backed VQE path. Install mindspore before using this feature."
        ) from exc

    if not hasattr(ms, runtime_config.mindspore_mode):
        raise ValueError(f"Unsupported MindSpore mode: {runtime_config.mindspore_mode}")

    mode = getattr(ms, runtime_config.mindspore_mode)
    ms.set_context(mode=mode, device_target=runtime_config.device_target)
    return {
        "mode": runtime_config.mindspore_mode,
        "device_target": runtime_config.device_target,
    }


def _require_framework_modules() -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import MindSpore and MindQuantum framework modules lazily."""
    try:
        import mindspore as ms
        from mindspore import Tensor, nn, ops
        from mindquantum.framework import MQAnsatzOnlyLayer, MQLayer
    except ImportError as exc:
        raise ImportError(
            "MindSpore and MindQuantum framework modules are required for full MindSpore VQE training."
        ) from exc
    return ms, Tensor, nn, ops, MQAnsatzOnlyLayer, MQLayer, np


def _build_vqe_objective_cell(ms: Any, ops: Any, layer: Any) -> Any:
    """Wrap the quantum layer so TrainOneStepCell sees a scalar objective."""
    class VQEAnsatzCell(ms.nn.Cell):
        def __init__(self, inner_layer: Any):
            super().__init__()
            self.inner_layer = inner_layer
            self.reduce_mean = ops.ReduceMean()

        def construct(self) -> Any:
            return self.reduce_mean(self.inner_layer())

    return VQEAnsatzCell(layer)


def _to_scalar(value: Any) -> float:
    """Convert MindSpore tensors or array-like values into a Python float."""
    if hasattr(value, "asnumpy"):
        array = np.asarray(value.asnumpy())
    else:
        array = np.asarray(value)
    return float(np.real(array.reshape(-1)[0]))


def _extract_grad_norm(train_output: Any) -> Optional[float]:
    """Estimate the gradient norm when TrainOneStepCell returns gradients."""
    if not isinstance(train_output, tuple) or len(train_output) != 2:
        return None
    _, grad_payload = train_output
    if isinstance(grad_payload, dict):
        squared_sum = 0.0
        for grad in grad_payload.values():
            grad_array = np.asarray(grad.asnumpy() if hasattr(grad, "asnumpy") else grad)
            squared_sum += float(np.sum(np.square(np.real(grad_array))))
        return math.sqrt(squared_sum)
    return None


def _current_layer_snapshot(layer: Any) -> MindSporeLayerSnapshot:
    """Read the current energy and trainable weights from a quantum layer."""
    energy = _to_scalar(layer())
    weights = np.asarray(layer.weight.asnumpy(), dtype=float).copy()
    return MindSporeLayerSnapshot(energy=energy, weights=weights)


def build_mindspore_training_cell(
    config: VQEConfig,
    control_value: float,
    initial_params: Optional[np.ndarray] = None,
) -> MindSporeVQEArtifacts:
    """Create the simulator, quantum layer, optimizer, and train cell for one point."""
    runtime_context = initialize_mindspore(config.runtime)
    ms, Tensor, nn, ops, MQAnsatzOnlyLayer, _, _ = _require_framework_modules()

    circuit_info = build_circuit(config.system)
    hamiltonian, _ = build_hamiltonian(config.system, control_value)

    try:
        from mindquantum.simulator import Simulator
    except ImportError as exc:
        raise ImportError(
            "MindQuantum simulator is required for MindSpore-backed VQE training."
        ) from exc

    simulator = Simulator(config.runtime.backend, config.system.n_qubits)
    grad_ops = simulator.get_expectation_with_grad(hamiltonian, circuit_info.circuit)

    init_params = (
        np.asarray(initial_params, dtype=np.float32).copy()
        if initial_params is not None
        else initialize_params(circuit_info.n_params, config.optimizer).astype(np.float32)
    )
    if init_params.shape[0] != circuit_info.n_params:
        raise ValueError("Initial parameter size does not match circuit parameter count.")

    layer = MQAnsatzOnlyLayer(grad_ops, weight=Tensor(init_params))
    objective = _build_vqe_objective_cell(ms, ops, layer)
    optimizer = nn.Adam(
        objective.trainable_params(),
        learning_rate=config.optimizer.learning_rate,
        beta1=config.optimizer.beta1,
        beta2=config.optimizer.beta2,
        eps=config.optimizer.epsilon,
    )
    try:
        train_cell = nn.TrainOneStepCell(objective, optimizer, return_grad=True)
        supports_return_grad = True
    except TypeError:
        train_cell = nn.TrainOneStepCell(objective, optimizer)
        supports_return_grad = False

    objective.set_train()
    train_cell.set_train()

    return MindSporeVQEArtifacts(
        grad_ops=grad_ops,
        layer=layer,
        objective=objective,
        optimizer=optimizer,
        train_cell=train_cell,
        circuit_info=circuit_info,
        supports_return_grad=supports_return_grad,
        runtime_context=runtime_context,
    )


def build_mindspore_inference_layer(
    config: VQEConfig,
    control_value: float,
    initial_params: Optional[np.ndarray] = None,
) -> MindSporeVQEArtifacts:
    """Alias training-cell construction for evaluation-only callers."""
    return build_mindspore_training_cell(
        config=config,
        control_value=control_value,
        initial_params=initial_params,
    )


def run_train_step(artifacts: MindSporeVQEArtifacts) -> MindSporeTrainStepResult:
    """Execute one optimization step and normalize the return payload."""
    train_output = artifacts.train_cell()
    if isinstance(train_output, tuple):
        loss_tensor = train_output[0]
    else:
        loss_tensor = train_output
    return MindSporeTrainStepResult(
        loss=_to_scalar(loss_tensor),
        grad_norm=_extract_grad_norm(train_output),
    )


def evaluate_mindspore_layer(artifacts: MindSporeVQEArtifacts) -> MindSporeLayerSnapshot:
    """Evaluate the current layer weights without permanently switching train mode."""
    artifacts.layer.set_train(False)
    snapshot = _current_layer_snapshot(artifacts.layer)
    artifacts.layer.set_train(True)
    return snapshot


def run_mindspore_single_point(
    config: VQEConfig,
    control_value: float,
    initial_params: Optional[np.ndarray] = None,
    init_source: str = "mindspore_fresh",
) -> PointResult:
    """Optimize one sweep point through MindSpore's training loop."""
    config.validate()
    artifacts = build_mindspore_training_cell(config, control_value, initial_params=initial_params)

    best_energy = math.inf
    best_params: Optional[np.ndarray] = None
    history: List[float] = []
    grad_norm_history: List[float] = []
    previous_energy = math.inf
    patience_count = 0
    status = "max_iters_reached"
    start_time = time.time()

    for _ in range(config.optimizer.max_iters):
        step_result = run_train_step(artifacts)
        snapshot = evaluate_mindspore_layer(artifacts)
        history.append(snapshot.energy)
        grad_norm_history.append(float(step_result.grad_norm) if step_result.grad_norm is not None else float("nan"))

        if snapshot.energy < best_energy:
            best_energy = snapshot.energy
            best_params = snapshot.weights.copy()

        if abs(snapshot.energy - previous_energy) < config.optimizer.tol:
            patience_count += 1
        else:
            patience_count = 0
        previous_energy = snapshot.energy

        if patience_count >= config.optimizer.patience:
            status = "converged"
            break

    final_snapshot = evaluate_mindspore_layer(artifacts)
    if best_params is None:
        best_params = final_snapshot.weights.copy()
        best_energy = final_snapshot.energy

    reference_energy = exact_reference_energy(config, control_value)
    energy_gap = None if reference_energy is None else float(abs(reference_energy - best_energy))

    return PointResult(
        lambda_value=float(control_value),
        status=status,
        converged=status == "converged",
        best_energy=float(best_energy),
        final_energy=float(final_snapshot.energy),
        n_iters=len(history),
        best_params=best_params.astype(float).tolist(),
        param_names=list(artifacts.circuit_info.param_names),
        history=history,
        grad_norm_history=grad_norm_history,
        init_source=init_source,
        reference_energy=reference_energy,
        energy_gap=energy_gap,
        metadata={
            "ansatz": artifacts.circuit_info.ansatz,
            "n_qubits": config.system.n_qubits,
            "depth": config.system.depth,
            "hamiltonian": config.system.hamiltonian,
            "backend": config.runtime.backend,
            "mindspore": artifacts.runtime_context,
            "supports_return_grad": artifacts.supports_return_grad,
            "elapsed_seconds": time.time() - start_time,
            **hamiltonian_metadata(config.system, control_value),
        },
    )


def run_mindspore_sweep(config: VQEConfig) -> SweepResult:
    """Run the full control-parameter sweep on the MindSpore-backed path."""
    config.validate()
    values = sweep_values(config.sweep)
    points: List[PointResult] = []
    next_initial: Optional[np.ndarray] = None

    for control_value in values:
        if config.sweep.warm_start and next_initial is not None:
            current_initial = next_initial.copy()
            init_source = "mindspore_warm_start"
        else:
            current_initial = None
            init_source = "mindspore_fresh"

        point = run_mindspore_single_point(
            config=config,
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
            "mode": "scan",
            "backend": config.runtime.backend,
            "warm_start": config.sweep.warm_start,
            "n_points": len(points),
            "lambda_values": values,
            "path": "mindspore",
        },
    )
