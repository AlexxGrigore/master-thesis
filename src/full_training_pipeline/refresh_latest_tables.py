from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from full_training_pipeline.plotting import plot_baseline_vs_corrected_metrics


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_markdown_table(handle, title: str, headers: list[str], rows: list[list[str]]) -> None:
    handle.write(f"## {title}\n\n")
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(row) + " |\n")
    handle.write("\n")


def _refresh_run_table(run_dir: Path) -> dict[str, object]:
    json_dir = run_dir / "json"
    plots_dir = run_dir / "plots"

    config = _load_json(json_dir / "config.json")
    training_summary = _load_json(run_dir / "training_summary.json")
    validation_baseline = _load_json(json_dir / "validation_baseline_metrics.json")
    validation_best = _load_json(json_dir / "validation_corrected_metrics.json")
    validation_last = _load_json(json_dir / "validation_corrected_metrics_last_epoch.json")
    test_baseline = _load_json(json_dir / "test_baseline_metrics.json")
    test_best = _load_json(json_dir / "test_corrected_metrics.json")
    test_last = _load_json(json_dir / "test_corrected_metrics_last_epoch.json")

    plot_baseline_vs_corrected_metrics(
        validation_baseline_metrics=validation_baseline,
        validation_best_metrics=validation_best,
        validation_last_metrics=validation_last,
        test_baseline_metrics=test_baseline,
        test_best_metrics=test_best,
        test_last_metrics=test_last,
        output_path=plots_dir / "baseline_vs_corrected_metrics.png",
    )

    return {
        "run_name": run_dir.name,
        "dataset_type": str(config["dataset_type"]),
        "model_type": str(config["model_type"]),
        "best_epoch": int(training_summary["best_epoch"]),
        "validation_baseline": float(validation_baseline["mean_focal_spot_error_mrad"]),
        "validation_best": float(validation_best["mean_focal_spot_error_mrad"]),
        "validation_median_baseline": float(validation_baseline["median_focal_spot_error_mrad"]),
        "validation_median_best": float(validation_best["median_focal_spot_error_mrad"]),
        "test_baseline": float(test_baseline["mean_focal_spot_error_mrad"]),
        "test_best": float(test_best["mean_focal_spot_error_mrad"]),
        "test_median_baseline": float(test_baseline["median_focal_spot_error_mrad"]),
        "test_median_best": float(test_best["median_focal_spot_error_mrad"]),
    }


def _sort_key(record: dict[str, object]) -> tuple[int, int, str]:
    dataset_order = {"synthetic": 0, "real": 1}
    model_order = {"linear": 0, "poly2": 1, "poly3": 2, "poly4": 3, "snn": 4}
    return (
        dataset_order.get(str(record["dataset_type"]), 99),
        model_order.get(str(record["model_type"]), 99),
        str(record["run_name"]),
    )


def build_tables(latest_dir: Path) -> None:
    run_dirs = [
        path
        for path in latest_dir.iterdir()
        if path.is_dir() and path.name != "old" and path.name.startswith("full_training_pipeline_")
    ]
    records = [_refresh_run_table(run_dir) for run_dir in sorted(run_dirs)]
    records.sort(key=_sort_key)

    comparison_headers = [
        "Dataset",
        "Model",
        "Run",
        "Best epoch",
        "Val baseline mean",
        "Val best mean",
        "Val improvement",
        "Test baseline mean",
        "Test best mean",
        "Test improvement",
    ]
    comparison_rows = [comparison_headers]
    comparison_md_rows: list[list[str]] = []

    baseline_headers = [
        "Dataset",
        "Model",
        "Val mean baseline",
        "Val mean best",
        "Val median baseline",
        "Val median best",
        "Test mean baseline",
        "Test mean best",
        "Test median baseline",
        "Test median best",
    ]
    baseline_rows = [baseline_headers]
    baseline_md_rows: list[list[str]] = []

    for record in records:
        val_improvement = float(record["validation_baseline"]) - float(record["validation_best"])
        test_improvement = float(record["test_baseline"]) - float(record["test_best"])

        comparison_row = [
            str(record["dataset_type"]),
            str(record["model_type"]),
            str(record["run_name"]),
            str(record["best_epoch"]),
            _format_float(float(record["validation_baseline"])),
            _format_float(float(record["validation_best"])),
            _format_float(val_improvement),
            _format_float(float(record["test_baseline"])),
            _format_float(float(record["test_best"])),
            _format_float(test_improvement),
        ]
        comparison_rows.append(comparison_row)
        comparison_md_rows.append(comparison_row)

        baseline_row = [
            str(record["dataset_type"]),
            str(record["model_type"]),
            _format_float(float(record["validation_baseline"])),
            _format_float(float(record["validation_best"])),
            _format_float(float(record["validation_median_baseline"])),
            _format_float(float(record["validation_median_best"])),
            _format_float(float(record["test_baseline"])),
            _format_float(float(record["test_best"])),
            _format_float(float(record["test_median_baseline"])),
            _format_float(float(record["test_median_best"])),
        ]
        baseline_rows.append(baseline_row)
        baseline_md_rows.append(baseline_row)

    _write_csv(latest_dir / "comparison_table.csv", comparison_rows)
    _write_csv(latest_dir / "baseline_best_metrics_table.csv", baseline_rows)

    markdown_path = latest_dir / "comparison_tables.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Latest Full Training Pipeline Comparison Tables\n\n")
        _write_markdown_table(handle, "Experiment Comparison", comparison_headers, comparison_md_rows)
        _write_markdown_table(handle, "Baseline vs Best Metrics", baseline_headers, baseline_md_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh latest run tables and summary exports.")
    parser.add_argument(
        "latest_dir",
        nargs="?",
        default=Path(__file__).resolve().parents[2] / "outputs" / "latest",
        type=Path,
        help="Path to the outputs/latest directory.",
    )
    args = parser.parse_args()
    build_tables(args.latest_dir.resolve())


if __name__ == "__main__":
    main()