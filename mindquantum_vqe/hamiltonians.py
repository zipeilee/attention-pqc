from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .config import SystemConfig

PauliTerm = Tuple[Tuple[int, str], ...]
WeightedPauliTerm = Tuple[float, PauliTerm]


def pair_edges(n_qubits: int, periodic: bool = True) -> List[Tuple[int, int]]:
    limit = n_qubits if periodic else n_qubits - 1
    return [(index, (index + 1) % n_qubits) for index in range(limit)]


def cluster_triplets(n_qubits: int, periodic: bool = True) -> List[Tuple[int, int, int]]:
    if periodic:
        return [((index - 1) % n_qubits, index, (index + 1) % n_qubits) for index in range(n_qubits)]
    return [(index, index + 1, index + 2) for index in range(n_qubits - 2)]


def _normalized_term(operators: Sequence[Tuple[int, str]]) -> PauliTerm:
    return tuple(sorted(operators, key=lambda item: item[0]))


def transverse_ising_terms(n_qubits: int, field: float, periodic: bool = True) -> List[WeightedPauliTerm]:
    terms: List[WeightedPauliTerm] = []
    for left, right in pair_edges(n_qubits, periodic=periodic):
        terms.append((-1.0, _normalized_term(((left, "Z"), (right, "Z")))))
    for qubit in range(n_qubits):
        terms.append((-float(field), ((qubit, "X"),)))
    return terms


def cluster_ising_2_terms(n_qubits: int, lam: float, periodic: bool = True) -> List[WeightedPauliTerm]:
    terms: List[WeightedPauliTerm] = []
    for left, center, right in cluster_triplets(n_qubits, periodic=periodic):
        terms.append((-1.0, _normalized_term(((left, "X"), (center, "Z"), (right, "X")))))
    for left, right in pair_edges(n_qubits, periodic=periodic):
        terms.append((float(lam), _normalized_term(((left, "Y"), (right, "Y")))))
    return terms


def cluster_ising_3_terms(n_qubits: int, h2: float) -> List[WeightedPauliTerm]:
    terms: List[WeightedPauliTerm] = []
    for index in range(n_qubits - 2):
        terms.append((-1.0, _normalized_term(((index, "Z"), (index + 1, "X"), (index + 2, "Z")))))
    for index in range(n_qubits - 1):
        terms.append((-float(h2), _normalized_term(((index, "X"), (index + 1, "X")))))
    for index in range(n_qubits):
        terms.append((-1.0, ((index, "X"),)))
    return terms


def hamiltonian_terms(system: SystemConfig, control_value: float) -> List[WeightedPauliTerm]:
    if system.hamiltonian == "transverse_ising":
        return transverse_ising_terms(system.n_qubits, control_value, periodic=system.periodic)
    if system.hamiltonian == "cluster_ising_2":
        return cluster_ising_2_terms(system.n_qubits, control_value, periodic=system.periodic)
    if system.hamiltonian == "cluster_ising_3":
        return cluster_ising_3_terms(system.n_qubits, control_value)
    raise ValueError(f"Unsupported hamiltonian: {system.hamiltonian}")


def term_to_mindquantum_string(term: PauliTerm) -> str:
    return " ".join(f"{pauli}{index}" for index, pauli in term)


def build_hamiltonian(system: SystemConfig, control_value: float) -> Tuple[Any, List[WeightedPauliTerm]]:
    try:
        from mindquantum.core.operators import Hamiltonian, QubitOperator
    except ImportError as exc:
        raise ImportError(
            "MindQuantum is required to construct Hamiltonians. Install mindquantum before running VQE."
        ) from exc

    terms = hamiltonian_terms(system, control_value)
    operator = QubitOperator()
    for coefficient, term in terms:
        operator += QubitOperator(term_to_mindquantum_string(term), coefficient)
    return Hamiltonian(operator), terms


def hamiltonian_metadata(system: SystemConfig, control_value: float) -> Dict[str, Any]:
    terms = hamiltonian_terms(system, control_value)
    return {
        "hamiltonian": system.hamiltonian,
        "control_value": control_value,
        "n_terms": len(terms),
        "periodic": system.periodic,
    }
