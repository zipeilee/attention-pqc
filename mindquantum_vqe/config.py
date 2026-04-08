from __future__ import annotations

"""Configuration models for MindQuantum/MindSpore VQE runs."""

from dataclasses import asdict, dataclass, field
from importlib import metadata
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 11)
RECOMMENDED_MINDQUANTUM = "0.12.0"
RECOMMENDED_MINDSPORE = "2.8.0"


@dataclass
class SystemConfig:
    """Static problem definition shared by circuit and Hamiltonian builders."""

    n_qubits: int = 4
    depth: int = 2
    ansatz: str = "t"
    hamiltonian: str = "cluster_ising_2"
    periodic: bool = True

    def validate(self) -> None:
        """Validate system-level choices before building quantum objects."""
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2.")
        if self.depth < 1:
            raise ValueError("depth must be >= 1.")
        if self.ansatz not in {"t", "basic_rot_ent"}:
            raise ValueError(f"Unsupported ansatz: {self.ansatz}")
        if self.hamiltonian not in {"transverse_ising", "cluster_ising_2", "cluster_ising_3"}:
            raise ValueError(f"Unsupported hamiltonian: {self.hamiltonian}")


@dataclass
class SweepConfig:
    """Control how the external Hamiltonian parameter is sampled."""

    enabled: bool = True
    lambda_start: float = 0.0
    lambda_stop: float = 0.8
    lambda_step: float = 0.1
    lambda_values: List[float] = field(default_factory=list)
    single_lambda: float = 0.0
    warm_start: bool = True

    def validate(self) -> None:
        """Check that the sweep range or explicit sweep values are well formed."""
        if self.lambda_values:
            return
        if self.lambda_step <= 0:
            raise ValueError("lambda_step must be > 0.")
        if self.lambda_stop < self.lambda_start:
            raise ValueError("lambda_stop must be >= lambda_start.")


@dataclass
class OptimizerConfig:
    """Numerical optimizer settings for variational parameter updates."""

    kind: str = "adam"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iters: int = 2000
    tol: float = 1e-6
    patience: int = 50
    init_strategy: str = "ones"
    init_scale: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        """Reject unsupported optimizer options in the first implementation."""
        if self.kind != "adam":
            raise ValueError("Only adam optimizer is supported in v1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.max_iters < 1:
            raise ValueError("max_iters must be >= 1.")
        if self.tol <= 0:
            raise ValueError("tol must be > 0.")
        if self.patience < 1:
            raise ValueError("patience must be >= 1.")
        if self.init_strategy not in {"zeros", "ones", "random"}:
            raise ValueError(f"Unsupported init_strategy: {self.init_strategy}")


@dataclass
class RuntimeConfig:
    """Execution-mode settings for simulator backend and MindSpore runtime."""

    mode: str = "scan"
    backend: str = "mqvector"
    use_mindspore: bool = False
    device_target: str = "CPU"
    mindspore_mode: str = "PYNATIVE_MODE"
    log_every: int = 10
    exact_reference_max_qubits: int = 8

    def validate(self) -> None:
        """Ensure runtime options stay within the supported v1 surface."""
        if self.mode not in {"single", "scan"}:
            raise ValueError(f"Unsupported runtime mode: {self.mode}")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1.")
        if self.mindspore_mode != "PYNATIVE_MODE":
            raise ValueError("v1 only supports MindSpore PYNATIVE_MODE.")


@dataclass
class OutputConfig:
    """Output-path and serialization controls for experiment artifacts."""

    output_dir: str = "outputs/mindquantum_vqe"
    run_name: str = "default"
    export_point_files: bool = True
    export_wide_csv: bool = True
    export_summary: bool = True
    export_history: bool = True
    overwrite: bool = True

    def validate(self) -> None:
        """Ensure output locations can be resolved deterministically."""
        if not self.output_dir:
            raise ValueError("output_dir must not be empty.")
        if not self.run_name:
            raise ValueError("run_name must not be empty.")

    def resolved_output_dir(self) -> Path:
        """Return the final directory for one named experiment run."""
        return Path(self.output_dir) / self.run_name


@dataclass
class VQEConfig:
    """Top-level experiment configuration composed of nested sections."""

    system: SystemConfig = field(default_factory=SystemConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Validate the full configuration and cross-section constraints."""
        self.system.validate()
        self.sweep.validate()
        self.optimizer.validate()
        self.runtime.validate()
        self.output.validate()
        if self.runtime.mode == "single" and self.sweep.enabled and self.sweep.lambda_values:
            if len(self.sweep.lambda_values) > 1:
                raise ValueError("single mode cannot use multiple lambda_values.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the nested dataclass tree into plain Python objects."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VQEConfig":
        """Build a configuration object from a dictionary payload."""
        return cls(
            system=SystemConfig(**payload.get("system", {})),
            sweep=SweepConfig(**payload.get("sweep", {})),
            optimizer=OptimizerConfig(**payload.get("optimizer", {})),
            runtime=RuntimeConfig(**payload.get("runtime", {})),
            output=OutputConfig(**payload.get("output", {})),
        )

    @classmethod
    def from_json(cls, file_path: str | Path) -> "VQEConfig":
        """Load configuration sections from a JSON file on disk."""
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def write_json(self, file_path: str | Path) -> None:
        """Persist the current configuration as UTF-8 JSON."""
        Path(file_path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _safe_version(package_name: str) -> Optional[str]:
    """Return an installed package version, or None when unavailable."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_info() -> Dict[str, Any]:
    """Collect version metadata used in exported reports and diagnostics."""
    return {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_supported_range": {
            "min": ".".join(map(str, MIN_PYTHON)),
            "max": ".".join(map(str, MAX_PYTHON)),
        },
        "recommended_versions": {
            "mindquantum": RECOMMENDED_MINDQUANTUM,
            "mindspore": RECOMMENDED_MINDSPORE,
        },
        "installed_versions": {
            "mindquantum": _safe_version("mindquantum"),
            "mindspore": _safe_version("mindspore"),
        },
    }
