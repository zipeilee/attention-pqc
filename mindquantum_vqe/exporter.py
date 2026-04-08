from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .config import VQEConfig
from .solver import PointResult, SweepResult


def ensure_output_dir(config: VQEConfig) -> Path:
    output_dir = config.output.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_config_snapshot(config: VQEConfig, output_dir: Path) -> Path:
    file_path = output_dir / "config.snapshot.json"
    file_path.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def export_point_files(points: Iterable[PointResult], output_dir: Path) -> List[Path]:
    exported: List[Path] = []
    for point in points:
        file_path = output_dir / f"{point.lambda_value:.3f}.csv"
        with file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["column1"])
            for value in point.best_params:
                writer.writerow([value])
        exported.append(file_path)
    return exported


def export_wide_params(points: Iterable[PointResult], output_dir: Path) -> Path:
    points = list(points)
    if not points:
        raise ValueError("No points available to export.")
    max_params = max(len(point.best_params) for point in points)
    file_path = output_dir / "params_wide.csv"
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["lambda_value"] + [f"param_{idx}" for idx in range(max_params)]
        writer.writerow(header)
        for point in points:
            row = [point.lambda_value] + list(point.best_params)
            row.extend([""] * (max_params - len(point.best_params)))
            writer.writerow(row)
    return file_path


def export_summary(points: Iterable[PointResult], output_dir: Path) -> Path:
    file_path = output_dir / "summary.csv"
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "lambda_value",
            "status",
            "converged",
            "best_energy",
            "final_energy",
            "reference_energy",
            "energy_gap",
            "n_iters",
            "init_source",
        ])
        for point in points:
            writer.writerow([
                point.lambda_value,
                point.status,
                point.converged,
                point.best_energy,
                point.final_energy,
                point.reference_energy,
                point.energy_gap,
                point.n_iters,
                point.init_source,
            ])
    return file_path


def export_histories(points: Iterable[PointResult], output_dir: Path) -> List[Path]:
    exported: List[Path] = []
    for point in points:
        file_path = output_dir / f"history_{point.lambda_value:.3f}.json"
        file_path.write_text(
            json.dumps(
                {
                    "lambda_value": point.lambda_value,
                    "history": point.history,
                    "grad_norm_history": point.grad_norm_history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        exported.append(file_path)
    return exported


def export_result_bundle(config: VQEConfig, result: SweepResult) -> Dict[str, str]:
    output_dir = ensure_output_dir(config)
    paths: Dict[str, str] = {}
    paths["config"] = str(write_config_snapshot(config, output_dir))
    if config.output.export_point_files:
        export_point_files(result.points, output_dir)
    if config.output.export_wide_csv:
        paths["params_wide"] = str(export_wide_params(result.points, output_dir))
    if config.output.export_summary:
        paths["summary"] = str(export_summary(result.points, output_dir))
    if config.output.export_history:
        export_histories(result.points, output_dir)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    paths["result"] = str(result_path)
    return paths
