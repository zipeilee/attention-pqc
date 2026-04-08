from __future__ import annotations

"""Top-level orchestration entry points for running VQE experiments."""

import argparse
import json
from typing import Any, Dict

from .config import VQEConfig, collect_environment_info
from .exporter import export_result_bundle
from .ms_adapter import run_mindspore_single_point, run_mindspore_sweep
from .solver import SweepResult, solve_single_point, solve_sweep
from .validation import build_validation_report


def run(config: VQEConfig) -> Dict[str, Any]:
    """Execute one configured experiment and export its result artifacts."""
    config.validate()
    if config.runtime.mode == "single":
        control_value = (
            config.sweep.lambda_values[0]
            if config.sweep.lambda_values
            else config.sweep.single_lambda
        )
        if config.runtime.use_mindspore:
            point = run_mindspore_single_point(config, control_value)
        else:
            point = solve_single_point(config, control_value)
        result = SweepResult(points=[point], metadata={"mode": "single"})
    else:
        if config.runtime.use_mindspore:
            result = run_mindspore_sweep(config)
        else:
            result = solve_sweep(config)

    exported = export_result_bundle(config, result)
    validation_value = result.points[0].lambda_value if result.points else 0.0
    validation_report = build_validation_report(config, validation_value)
    output_dir = config.output.resolved_output_dir()
    validation_path = output_dir / "validation.json"
    validation_path.write_text(json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "result": result.to_dict(),
        "exported": exported,
        "validation": str(validation_path),
        "environment": collect_environment_info(),
    }


def parse_args() -> argparse.Namespace:
    """Parse the minimal CLI used to launch a JSON-configured experiment."""
    parser = argparse.ArgumentParser(description="Run MindQuantum VQE experiments.")
    parser.add_argument("--config", type=str, required=False, help="Path to a JSON config file.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point that loads configuration and prints a JSON report."""
    args = parse_args()
    config = VQEConfig.from_json(args.config) if args.config else VQEConfig()
    report = run(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
