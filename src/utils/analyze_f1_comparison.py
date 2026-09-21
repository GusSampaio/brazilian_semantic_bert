import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare test F1 by semantic role across trained models."
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
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

        for metric_name, value in metrics.items():
            if not metric_name.startswith("test_f1_"):
                continue
            role = metric_name.removeprefix("test_f1_")
            discovered_roles.add(role)
            scores[model_name][role].append(float(value))

    return scores, sorted(discovered_roles, key=role_sort_key)


def save_scores_csv(path, scores, roles, strategy):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["model", "strategy", "role", "mean_f1", "std_f1", "seeds"])
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


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(
        args.output_dir or artifacts_dir / "comparisons" / args.strategy
    )

    scores, roles = load_scores(artifacts_dir, args.strategy)
    if not scores:
        raise FileNotFoundError(
            f"No per-role test metrics found for strategy '{args.strategy}' "
            f"under {artifacts_dir}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_scores_csv(
        output_dir / "f1_by_class_and_model.csv",
        scores,
        roles,
        args.strategy,
    )
    save_comparison_plot(
        output_dir / "f1_by_class_and_model.png",
        scores,
        roles,
        args.strategy,
    )
    print(f"Comparison saved to {output_dir}")


if __name__ == "__main__":
    main()
