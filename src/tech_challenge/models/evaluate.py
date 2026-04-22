from __future__ import annotations

import json
from pathlib import Path

from tech_challenge.config import Settings


def _write_markdown_table(rows: list[dict[str, object]]) -> str:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    settings = Settings()
    summary = {}
    for filename in ("baseline_results.json", "mlp_results.json", "threshold_analysis.json"):
        path = settings.reports_dir / filename
        if path.exists():
            summary[filename] = json.loads(path.read_text(encoding="utf-8"))
        else:
            summary[filename] = f"Missing file: {Path(path).name}"

    baseline_results = summary.get("baseline_results.json")
    mlp_results = summary.get("mlp_results.json")
    if isinstance(baseline_results, list) and isinstance(mlp_results, dict):
        comparison_rows = []
        for row in baseline_results:
            comparison_rows.append(
                {
                    "model": row["model"],
                    "pr_auc": row["pr_auc"],
                    "roc_auc": row["roc_auc"],
                    "f1": row["f1"],
                    "recall": row["recall"],
                    "precision": row["precision"],
                    "expected_cost_at_threshold": row["expected_cost_at_threshold"],
                    "best_threshold_by_cost": row["best_threshold_by_cost"],
                    "best_expected_cost": row["best_expected_cost"],
                }
            )
        comparison_rows.append(
            {
                "model": "mlp_pytorch",
                "pr_auc": mlp_results["pr_auc"],
                "roc_auc": mlp_results["roc_auc"],
                "f1": mlp_results["f1"],
                "recall": mlp_results["recall"],
                "precision": mlp_results["precision"],
                "expected_cost_at_threshold": mlp_results["expected_cost_at_threshold"],
                "best_threshold_by_cost": mlp_results["best_threshold_by_cost"],
                "best_expected_cost": mlp_results["best_expected_cost"],
            }
        )
        comparison_rows.sort(key=lambda row: row["pr_auc"], reverse=True)
        comparison_path = settings.reports_dir / "model_comparison.json"
        comparison_path.write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
        comparison_markdown = settings.reports_dir / "model_comparison.md"
        comparison_markdown.write_text(_write_markdown_table(comparison_rows), encoding="utf-8")
        summary["model_comparison.json"] = comparison_rows

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
