import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datasets import DatasetDict
from plotly.colors import hex_to_rgb, qualitative


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare test F1 by semantic role across trained models."
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--strategy", default="baseline")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def role_sort_key(role):
    if role.startswith("ARG") and role[3:].isdigit():
        return 0, int(role[3:])
    if role.startswith("ARGM-"):
        return 1, role
    return 2, role


def load_scores(artifacts_dir, strategy):
    scores = defaultdict(lambda: defaultdict(list))
    discovered_roles = set()

    metrics_pattern = f"*/{strategy}/seed*/final_metrics.json"
    for metrics_path in artifacts_dir.glob(metrics_pattern):
        model_name = metrics_path.relative_to(artifacts_dir).parts[0]
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

        role_metrics = {
            metric_name: value
            for metric_name, value in metrics.items()
            if metric_name.startswith("test_f1_")
        }
        if not role_metrics:
            print(
                f"Warning: ignoring {metrics_path}: no test_f1_* metrics found."
            )
            continue

        for metric_name, value in role_metrics.items():
            role = metric_name.removeprefix("test_f1_")
            discovered_roles.add(role)
            scores[model_name][role].append(float(value))

    return scores, sorted(discovered_roles, key=role_sort_key)


def load_label_support(data_dir):
    with (data_dir / "labels_and_ids" / "id2label.json").open(
        "r", encoding="utf-8"
    ) as labels_file:
        id2label = json.load(labels_file)

    datasets = DatasetDict.load_from_disk(str(data_dir / "data_splits"))
    support = {}
    for split in ("train", "validation", "test"):
        counts = Counter()
        for labels in datasets[split]["labels"]:
            for label_id in labels:
                label_id = int(label_id)
                if label_id != -100:
                    counts[id2label[str(label_id)]] += 1
        support[split] = counts

    return support


def save_scores_csv(path, scores, roles, strategy, support):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "model",
                "strategy",
                "role",
                "mean_f1",
                "std_f1",
                "seeds",
                "train_support",
                "validation_support",
                "test_support",
            ]
        )
        for model_name in sorted(scores):
            for role in roles:
                values = scores[model_name].get(role, [])
                if not values:
                    continue
                writer.writerow(
                    [
                        model_name,
                        strategy,
                        role,
                        float(np.mean(values)),
                        float(np.std(values)),
                        len(values),
                        support["train"].get(role, 0),
                        support["validation"].get(role, 0),
                        support["test"].get(role, 0),
                    ]
                )


def save_comparison_plot(path, scores, roles, strategy):
    x_positions = np.arange(len(roles))
    figure_width = max(14, 0.55 * len(roles))
    figure, axis = plt.subplots(figsize=(figure_width, 7))

    for model_name in sorted(scores):
        means = np.array(
            [
                np.mean(scores[model_name][role])
                if scores[model_name].get(role)
                else np.nan
                for role in roles
            ],
            dtype=float,
        )
        standard_deviations = np.array(
            [
                np.std(scores[model_name][role])
                if scores[model_name].get(role)
                else np.nan
                for role in roles
            ],
            dtype=float,
        )
        line = axis.plot(
            x_positions,
            means,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=model_name,
        )[0]
        axis.fill_between(
            x_positions,
            np.clip(means - standard_deviations, 0, 1),
            np.clip(means + standard_deviations, 0, 1),
            color=line.get_color(),
            alpha=0.12,
        )

    axis.set(
        xticks=x_positions,
        xticklabels=roles,
        ylim=(0, 1.03),
        xlabel="Semantic role",
        ylabel="Test F1",
        title=f"F1 by semantic role and model ({strategy}, mean across seeds)",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_interactive_plot(path, scores, roles, strategy, support):
    figure = go.Figure()

    for model_index, model_name in enumerate(sorted(scores)):
        color = qualitative.Plotly[model_index % len(qualitative.Plotly)]
        red, green, blue = hex_to_rgb(color)
        means = []
        standard_deviations = []
        seed_counts = []
        train_support = []
        validation_support = []
        test_support = []
        for role in roles:
            values = scores[model_name].get(role, [])
            means.append(float(np.mean(values)) if values else None)
            standard_deviations.append(float(np.std(values)) if values else None)
            seed_counts.append(len(values))
            train_support.append(support["train"].get(role, 0))
            validation_support.append(support["validation"].get(role, 0))
            test_support.append(support["test"].get(role, 0))

        lower_bounds = [
            max(0.0, mean - deviation)
            if mean is not None and deviation is not None
            else None
            for mean, deviation in zip(means, standard_deviations)
        ]
        upper_bounds = [
            min(1.0, mean + deviation)
            if mean is not None and deviation is not None
            else None
            for mean, deviation in zip(means, standard_deviations)
        ]

        figure.add_trace(
            go.Scatter(
                x=roles,
                y=lower_bounds,
                mode="lines",
                legendgroup=model_name,
                line={"width": 0, "color": color},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=roles,
                y=upper_bounds,
                mode="lines",
                legendgroup=model_name,
                line={"width": 0, "color": color},
                fill="tonexty",
                fillcolor=f"rgba({red}, {green}, {blue}, 0.15)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=roles,
                y=means,
                mode="lines+markers",
                name=model_name,
                legendgroup=model_name,
                line={"color": color},
                marker={"color": color},
                customdata=np.column_stack(
                    (
                        standard_deviations,
                        seed_counts,
                        train_support,
                        validation_support,
                        test_support,
                    )
                ),
                hovertemplate=(
                    "Model: %{fullData.name}<br>"
                    "Role: %{x}<br>"
                    "Mean F1: %{y:.4f}<br>"
                    "Std: %{customdata[0]:.4f}<br>"
                    "Seeds: %{customdata[1]:.0f}<br>"
                    "Train instances: %{customdata[2]:.0f}<br>"
                    "Validation instances: %{customdata[3]:.0f}<br>"
                    "Test instances: %{customdata[4]:.0f}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=f"F1 by semantic role and model ({strategy}, mean across seeds)",
        xaxis_title="Semantic role",
        yaxis_title="Test F1",
        yaxis={"range": [0, 1.03]},
        hovermode="closest",
        template="plotly_white",
        legend_title="Model",
        legend={"groupclick": "togglegroup"},
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(
        args.output_dir or artifacts_dir / "comparisons" / args.strategy
    )

    scores, roles = load_scores(artifacts_dir, args.strategy)
    if not scores:
        raise FileNotFoundError(
            f"No per-role test metrics found for strategy '{args.strategy}' "
            f"under {artifacts_dir}."
        )

    support = load_label_support(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_scores_csv(
        output_dir / "f1_by_class_and_model.csv",
        scores,
        roles,
        args.strategy,
        support,
    )
    save_comparison_plot(
        output_dir / "f1_by_class_and_model.png",
        scores,
        roles,
        args.strategy,
    )
    save_interactive_plot(
        output_dir / "f1_by_class_and_model.html",
        scores,
        roles,
        args.strategy,
        support,
    )
    print(f"Comparison saved to {output_dir}")


if __name__ == "__main__":
    main()
