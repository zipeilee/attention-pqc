from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .circuits import build_circuit
from .config import VQEConfig
from .hamiltonians import build_hamiltonian
from .solver import build_expectation_with_grad, exact_reference_energy


def validate_circuit_structure(config: VQEConfig) -> Dict[str, Any]:
    circuit_info = build_circuit(config.system)
    return {
        "ansatz": circuit_info.ansatz,
        "n_params": circuit_info.n_params,
        "param_names_stable": circuit_info.param_names == list(circuit_info.param_names),
    }


def validate_hamiltonian_structure(config: VQEConfig, control_value: float) -> Dict[str, Any]:
    _, terms = build_hamiltonian(config.system, control_value)
    return {
        "hamiltonian": config.system.hamiltonian,
        "control_value": control_value,
        "n_terms": len(terms),
        "terms_preview": [str(term) for term in terms[: min(5, len(terms))]],
    }


def finite_difference_gradient_check(
    config: VQEConfig,
    control_value: float,
    epsilon: float = 1e-6,
    n_checks: int = 3,
) -> Dict[str, Any]:
    circuit_info, _, evaluator = build_expectation_with_grad(config, control_value)
    params = np.ones(circuit_info.n_params, dtype=float)
    energy, grad = evaluator(params)
    checks: List[Dict[str, float]] = []
    for index in range(min(n_checks, circuit_info.n_params)):
        plus = params.copy()
        minus = params.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_energy, _ = evaluator(plus)
        minus_energy, _ = evaluator(minus)
        approx = (plus_energy - minus_energy) / (2 * epsilon)
        checks.append(
            {
                "index": index,
                "analytic": float(grad[index]),
                "finite_difference": float(approx),
                "abs_error": float(abs(grad[index] - approx)),
            }
        )
    return {
        "energy_at_test_point": float(energy),
        "checks": checks,
    }


def build_validation_report(config: VQEConfig, control_value: float) -> Dict[str, Any]:
    return {
        "circuit": validate_circuit_structure(config),
        "hamiltonian": validate_hamiltonian_structure(config, control_value),
        "reference_energy": exact_reference_energy(config, control_value),
        "gradient_check": finite_difference_gradient_check(config, control_value),
    }
