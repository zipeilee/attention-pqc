from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from .config import SystemConfig


@dataclass
class CircuitBuildResult:
    circuit: Any
    param_names: List[str]
    n_params: int
    ansatz: str


def _require_mindquantum_circuit_api() -> Tuple[Any, Any, Any, Any]:
    try:
        from mindquantum.core.circuit import Circuit
        from mindquantum.core.gates import CZ, RX, RZ
    except ImportError as exc:
        raise ImportError(
            "MindQuantum is required to construct circuits. Install mindquantum before running VQE."
        ) from exc
    return Circuit, RX, RZ, CZ


def even_edges(n_qubits: int) -> List[Tuple[int, int]]:
    return [(start, start + 1) for start in range(0, n_qubits - 1, 2)]


def odd_edges(n_qubits: int) -> List[Tuple[int, int]]:
    return [(start, start + 1) for start in range(1, n_qubits - 1, 2)]


def _append_rx_layer(circuit: Any, RX: Any, names: List[str], prefix: str, n_qubits: int) -> None:
    for qubit in range(n_qubits):
        name = f"{prefix}_q{qubit}_rx"
        circuit += RX(name).on(qubit)
        names.append(name)


def _append_rz_layer(circuit: Any, RZ: Any, names: List[str], prefix: str, n_qubits: int) -> None:
    for qubit in range(n_qubits):
        name = f"{prefix}_q{qubit}_rz"
        circuit += RZ(name).on(qubit)
        names.append(name)


def _append_cz_edges(circuit: Any, CZ: Any, edges: Sequence[Tuple[int, int]]) -> None:
    for left, right in edges:
        circuit += CZ.on(right, left)


def parameter_names_for_t(n_qubits: int, depth: int) -> List[str]:
    names: List[str] = []
    for layer in range(depth):
        for phase in ("even_rx1", "even_rz", "even_rx2", "odd_rx1", "odd_rz", "odd_rx2"):
            for qubit in range(n_qubits):
                gate = "rz" if "rz" in phase else "rx"
                names.append(f"theta_l{layer}_{phase}_q{qubit}_{gate}")
    return names


def parameter_names_for_basic(n_qubits: int, depth: int) -> List[str]:
    names: List[str] = []
    for layer in range(depth):
        for qubit in range(n_qubits):
            names.append(f"theta_l{layer}_rx_q{qubit}_rx")
        for qubit in range(n_qubits):
            names.append(f"theta_l{layer}_rz_q{qubit}_rz")
    return names


def build_basic_rot_ent_circuit(n_qubits: int, depth: int) -> CircuitBuildResult:
    Circuit, RX, RZ, CZ = _require_mindquantum_circuit_api()
    circuit = Circuit()
    param_names: List[str] = []
    for layer in range(depth):
        _append_rx_layer(circuit, RX, param_names, f"theta_l{layer}_rx", n_qubits)
        _append_rz_layer(circuit, RZ, param_names, f"theta_l{layer}_rz", n_qubits)
        _append_cz_edges(circuit, CZ, even_edges(n_qubits))
        _append_cz_edges(circuit, CZ, odd_edges(n_qubits))
    return CircuitBuildResult(
        circuit=circuit,
        param_names=param_names,
        n_params=len(param_names),
        ansatz="basic_rot_ent",
    )


def build_t_circuit(n_qubits: int, depth: int) -> CircuitBuildResult:
    Circuit, RX, RZ, CZ = _require_mindquantum_circuit_api()
    circuit = Circuit()
    param_names: List[str] = []
    for layer in range(depth):
        _append_rx_layer(circuit, RX, param_names, f"theta_l{layer}_even_rx1", n_qubits)
        _append_rz_layer(circuit, RZ, param_names, f"theta_l{layer}_even_rz", n_qubits)
        _append_rx_layer(circuit, RX, param_names, f"theta_l{layer}_even_rx2", n_qubits)
        _append_cz_edges(circuit, CZ, even_edges(n_qubits))

        _append_rx_layer(circuit, RX, param_names, f"theta_l{layer}_odd_rx1", n_qubits)
        _append_rz_layer(circuit, RZ, param_names, f"theta_l{layer}_odd_rz", n_qubits)
        _append_rx_layer(circuit, RX, param_names, f"theta_l{layer}_odd_rx2", n_qubits)
        _append_cz_edges(circuit, CZ, odd_edges(n_qubits))

    return CircuitBuildResult(
        circuit=circuit,
        param_names=param_names,
        n_params=len(param_names),
        ansatz="t",
    )


def build_circuit(system: SystemConfig) -> CircuitBuildResult:
    if system.ansatz == "t":
        return build_t_circuit(system.n_qubits, system.depth)
    if system.ansatz == "basic_rot_ent":
        return build_basic_rot_ent_circuit(system.n_qubits, system.depth)
    raise ValueError(f"Unsupported ansatz: {system.ansatz}")
