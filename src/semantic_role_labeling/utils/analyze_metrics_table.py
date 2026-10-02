import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("precision", "Precisão"),
    ("recall", "Recall"),
    ("f1", "F1"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG, CSV and LaTeX tables comparing test metrics across "
            "models, strategies and seeds."
        )
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to include. By default, includes every strategy found.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--title", default="Desempenho dos modelos no conjunto de teste")
    return parser.parse_args()


def load_results(artifacts_dir, strategies=None):
    grouped = defaultdict(lambda: defaultdict(list))
    seed_counts = defaultdict(int)
    selected_strategies = set(strategies) if strategies else None

    for metrics_path in sorted(artifacts_dir.glob("*/*/seed*/final_metrics.json")):
        relative_parts = metrics_path.relative_to(artifacts_dir).parts
        model_name, strategy = relative_parts[:2]
        if selected_strategies and strategy not in selected_strategies:
            continue

        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

        values = {}
        for metric_name, _ in METRICS:
            key = f"test_{metric_name}"
            if key not in metrics:
                print(f"Warning: ignoring {metrics_path}: missing '{key}'.")
                break
            values[metric_name] = float(metrics[key])
        else:
            group_key = (model_name, strategy)
            for metric_name, value in values.items():
                grouped[group_key][metric_name].append(value)
            seed_counts[group_key] += 1

    results = []
    for (model_name, strategy), metric_values in sorted(grouped.items()):
        results.append(
            {
                "model": model_name,
                "strategy": strategy,
                "seeds": seed_counts[(model_name, strategy)],
                **{
                    f"{metric_name}_{statistic}": float(function(metric_values[metric_name]))
                    for metric_name, _ in METRICS
                    for statistic, function in (("mean", np.mean), ("std", np.std))
                },
            }
        )
    return results


def format_score(result, metric_name, decimals, latex=False):
    mean = result[f"{metric_name}_mean"]
    standard_deviation = result[f"{metric_name}_std"]
    if result["seeds"] == 1:
        return f"{mean:.{decimals}f}"
    separator = r" $\pm$ " if latex else " ± "
    return f"{mean:.{decimals}f}{separator}{standard_deviation:.{decimals}f}"


def best_values(results):
    return {
        metric_name: max(result[f"{metric_name}_mean"] for result in results)
        for metric_name, _ in METRICS
    }


def save_csv(path, results):
    fieldnames = ["model", "strategy", "seeds"]
    for metric_name, _ in METRICS:
        fieldnames.extend((f"{metric_name}_mean", f"{metric_name}_std"))

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def escape_latex(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def display_strategy(strategy):
    return strategy.replace("_", " ").title()


def save_latex(path, results, title, decimals):
    best = best_values(results)
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        f"    \\caption{{{escape_latex(title)}}}",
        r"    \label{tab:model_metrics}",
        r"    \begin{tabular}{llccc}",
        r"        \toprule",
        r"        Modelo & Estratégia & Precisão & Recall & F1 \\",
        r"        \midrule",
    ]
    for result in results:
        scores = []
        for metric_name, _ in METRICS:
            score = format_score(result, metric_name, decimals, latex=True)
            if np.isclose(result[f"{metric_name}_mean"], best[metric_name]):
                score = rf"\textbf{{{score}}}"
            scores.append(score)
        lines.append(
            "        "
            + " & ".join(
                (
                    escape_latex(result["model"]),
                    escape_latex(display_strategy(result["strategy"])),
                    *scores,
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_png(path, results, title, decimals):
    headers = ("Modelo", "Estratégia", *(label for _, label in METRICS))
    column_edges = np.array((0.0, 0.34, 0.56, 0.72, 0.86, 1.0))
    column_centers = (column_edges[:-1] + column_edges[1:]) / 2
    row_height = 0.62
    figure_height = max(2.4, 1.4 + row_height * len(results))
    figure, axis = plt.subplots(figsize=(13, figure_height))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, len(results) + 1.55)
    axis.axis("off")

    best = best_values(results)
    header_y = len(results) + 0.72
    axis.text(
        0,
        len(results) + 1.32,
        title,
        ha="left",
        va="center",
        fontsize=17,
        fontweight="bold",
        fontfamily="DejaVu Serif",
    )
    axis.hlines(header_y + 0.43, 0, 1, color="#161616", linewidth=1.8)
    axis.hlines(header_y - 0.38, 0, 1, color="#161616", linewidth=1.0)

    for column_index, header in enumerate(headers):
        alignment = "left" if column_index < 2 else "center"
        x_position = column_edges[column_index] + 0.01 if column_index < 2 else column_centers[column_index]
        axis.text(
            x_position,
            header_y,
            header,
            ha=alignment,
            va="center",
            fontsize=11.5,
            fontweight="bold",
            fontfamily="DejaVu Serif",
        )

    for row_index, result in enumerate(results):
        y_position = len(results) - row_index - 0.02
        if row_index % 2:
            axis.axhspan(y_position - 0.31, y_position + 0.31, color="#f3f3f3", zorder=0)
        axis.text(0.01, y_position, result["model"], ha="left", va="center", fontsize=10.5)
        axis.text(
            column_edges[1] + 0.01,
            y_position,
            display_strategy(result["strategy"]),
            ha="left",
            va="center",
            fontsize=10.5,
        )
        for metric_index, (metric_name, _) in enumerate(METRICS, start=2):
            is_best = np.isclose(result[f"{metric_name}_mean"], best[metric_name])
            axis.text(
                column_centers[metric_index],
                y_position,
                format_score(result, metric_name, decimals),
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold" if is_best else "normal",
            )

    axis.hlines(0.48, 0, 1, color="#161616", linewidth=1.8)
    figure.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(
        args.output_dir or artifacts_dir / "comparisons" / "metrics_table"
    )
    results = load_results(artifacts_dir, args.strategies)
    if not results:
        strategy_message = f" for strategies {args.strategies}" if args.strategies else ""
        raise FileNotFoundError(
            f"No complete test metrics found{strategy_message} under {artifacts_dir}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "model_metrics.csv", results)
    save_latex(
        output_dir / "model_metrics.tex", results, args.title, args.decimals
    )
    save_png(
        output_dir / "model_metrics.png", results, args.title, args.decimals
    )
    print(f"Metrics table saved to {output_dir} ({len(results)} rows).")


if __name__ == "__main__":
    main()