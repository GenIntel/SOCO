#!/usr/bin/env python3
"""Evaluate SOCO lvlm result workbooks.

For every model and modality, this script reports:

* average accuracy across all workbook rows; and
* circular accuracy, where a question group is correct only when every row in
  that group is correct.

Only the latest timestamped workbook for each model/modality is evaluated.

Example CLI usage::

    python evaluate_soco_c.py
    python evaluate_soco_c.py --base outputs --out-dir reports
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# The modality keys are also used in output column names.
DATASETS = {
    "img": "soco_lvlm_img_c",
    "imgtxt": "soco_lvlm_imgtxt_c",
    "txt": "soco_lvlm_txt_c",
}
MODALITIES = tuple(DATASETS)
METRIC_COLUMNS = tuple(
    f"{metric}_{modality}"
    for metric in ("avg", "circular")
    for modality in MODALITIES
)
CSV_COLUMNS = ("model", *METRIC_COLUMNS, *(f"path_{x}" for x in MODALITIES))


@dataclass(frozen=True)
class Metrics:
    """The scores and source file for one model/modality."""

    path: Path
    correct_rows: int
    total_rows: int
    correct_groups: int
    total_groups: int

    @property
    def average_accuracy(self) -> float:
        return self.correct_rows / self.total_rows if self.total_rows else 0.0

    @property
    def circular_accuracy(self) -> float:
        return self.correct_groups / self.total_groups if self.total_groups else 0.0


def read_workbook(path: Path) -> list[dict[str, object]]:
    """Read the first worksheet and return one dictionary per data row."""
    dataframe = pd.read_excel(path, sheet_name=0)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    # Pandas represents empty spreadsheet cells as NaN. Convert them back to
    # empty strings so answer parsing does not mistake "nan" for model output.
    return dataframe.fillna("").to_dict(orient="records")


def parse_prediction(value: object) -> str:
    """Extract an A-D answer using the conventions found in model outputs."""
    text = "" if value is None else str(value)
    answer = r"([A-D])"

    # Explicit final-answer forms are the most reliable, so prefer the last one.
    for pattern in (
        rf"\\boxed\s*{{\s*{answer}\s*}}",
        rf"final\s+answer\s*(?:is|:)?\s*\$?\s*{answer}(?![A-Za-z0-9])",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    standalone = re.findall(
        rf"(?<![A-Za-z0-9]){answer}(?![A-Za-z0-9])", text, flags=re.IGNORECASE
    )
    distinct = list(dict.fromkeys(match.upper() for match in standalone))
    if len(distinct) == 1:
        return distinct[0]

    # When several letters occur, use the formats emitted by older evaluators.
    for pattern in (
        rf"{answer}\s*[:.]\s*Point\s*[A-D]",
        rf"The\s+correct\s+answer\s+is\s*{answer}",
        rf"^\s*{answer}(?![A-Za-z0-9])",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return ""


def evaluate_workbook(path: Path) -> Metrics:
    """Calculate row-level average and group-level circular accuracy."""
    rows = read_workbook(path)
    required_columns = {"answer", "g_index", "prediction"}
    columns = set(rows[0]) if rows else set()
    missing = required_columns - columns
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    results_by_group: dict[str, list[bool]] = defaultdict(list)
    correct_rows = 0
    for row in rows:
        answer = str(row["answer"]).strip().upper()
        is_correct = parse_prediction(row["prediction"]) == answer
        correct_rows += is_correct
        results_by_group[str(row["g_index"]).strip()].append(is_correct)

    correct_groups = sum(all(results) for results in results_by_group.values())
    return Metrics(path, correct_rows, len(rows), correct_groups, len(results_by_group))


def timestamp_key(path: Path) -> tuple[int, str]:
    """Return a sortable key for parent directories such as ``T123_name``."""
    match = re.match(r"T(\d+)_", path.parent.name)
    return (int(match.group(1)) if match else -1, str(path))


def discover_workbooks(base: Path) -> dict[str, dict[str, Path]]:
    """Find the latest workbook for each modality and model."""
    found: dict[str, dict[str, Path]] = {modality: {} for modality in MODALITIES}
    for path in base.glob("*/*/*.xlsx"):
        if path.name.endswith("_openai_result.xlsx") or any(
            part.startswith("bak_") for part in path.parts
        ):
            continue

        for modality, dataset in DATASETS.items():
            suffix = f"_{dataset}.xlsx"
            if not path.name.endswith(suffix):
                continue

            model = path.name[: -len(suffix)]
            previous = found[modality].get(model)
            if previous is None or timestamp_key(path) > timestamp_key(previous):
                found[modality][model] = path
            break
    return found


def evaluate_all(base: Path) -> dict[str, dict[str, Metrics]]:
    """Discover and evaluate every available model/modality pair."""
    return {
        modality: {
            model: evaluate_workbook(path)
            for model, path in sorted(workbooks.items())
        }
        for modality, workbooks in discover_workbooks(base).items()
    }


def percentage(value: float) -> str:
    return f"{value * 100:.1f}"


def build_rows(metrics: dict[str, dict[str, Metrics]], base: Path) -> list[dict[str, object]]:
    """Flatten results into the shape shared by the CSV and Markdown reports."""
    models = sorted({model for results in metrics.values() for model in results})
    output = []
    for model in models:
        row: dict[str, object] = {"model": model}
        for modality in MODALITIES:
            result = metrics[modality].get(model)
            row[f"avg_{modality}"] = percentage(result.average_accuracy) if result else ""
            row[f"circular_{modality}"] = (
                percentage(result.circular_accuracy) if result else ""
            )
            row[f"path_{modality}"] = str(result.path.relative_to(base)) if result else ""
        output.append(row)
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# soco lvlm c",
        "",
        "Values are percentages. The latest timestamped `c` workbook is used for each model and modality.",
        "",
        "| Model | Avg I | Avg I+T | Avg T | Circular I | Circular I+T | Circular T |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [str(row[column] or "--") for column in METRIC_COLUMNS]
        lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
    return "\n".join((*lines, ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="VLMEvalKit outputs directory to scan (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Report directory (default: the --base directory).",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    out_dir = (args.out_dir or base).resolve()
    metrics = evaluate_all(base)
    if not any(metrics.values()):
        raise SystemExit("No soco_lvlm result files found.")

    rows = build_rows(metrics, base)
    report = make_markdown(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "soco_lvlm_c.md"
    csv_path = out_dir / "soco_lvlm_c.csv"
    markdown_path.write_text(report, encoding="utf-8")
    write_csv(rows, csv_path)

    print(f"Wrote {markdown_path}\nWrote {csv_path}\n\n{report.rstrip()}")


if __name__ == "__main__":
    main()
