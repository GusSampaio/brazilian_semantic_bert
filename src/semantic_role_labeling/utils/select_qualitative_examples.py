import argparse
import json
from pathlib import Path


DEFAULT_INPUT = (
    "artifacts/xlm-roberta-large/baseline/seed120/"
    "confusion_analysis/instance_analysis.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select representative SRL successes and failures."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def is_argument(role):
    return role not in {None, "O", "PRED"}


def is_numbered(role):
    return role is not None and role.startswith("ARG") and not role.startswith("ARGM")


def is_modifier(role):
    return role is not None and role.startswith("ARGM")


def evaluated_tokens(instance):
    return [
        token
        for token in instance["model_subtokens"]
        if not token["ignored_in_evaluation"]
    ]


def true_arguments(instance):
    return [
        token for token in evaluated_tokens(instance) if is_argument(token["true_role"])
    ]


def all_family_arguments_correct(instance, family_check):
    family_arguments = [
        token for token in true_arguments(instance) if family_check(token["true_role"])
    ]
    return bool(family_arguments) and all(token["correct"] for token in family_arguments)


def all_arguments_wrong(instance):
    arguments = true_arguments(instance)
    return bool(arguments) and all(not token["correct"] for token in arguments)


def family_summary(instance, family_check):
    tokens = evaluated_tokens(instance)
    true_family = [token for token in tokens if family_check(token["true_role"])]
    false_positives = [
        token
        for token in tokens
        if family_check(token["predicted_role"])
        and not family_check(token["true_role"])
    ]
    return {
        "true_family_arguments": len(true_family),
        "correct_family_arguments": sum(token["correct"] for token in true_family),
        "false_positive_family_arguments": len(false_positives),
    }


def compact_instance(instance, category, family_check=None):
    tokens = evaluated_tokens(instance)
    arguments = [
        {
            "token": token["source_token"],
            "subtoken": token["subtoken"],
            "true_role": token["true_role"],
            "predicted_role": token["predicted_role"],
            "confidence": token["confidence"],
            "correct": token["correct"],
        }
        for token in tokens
        if is_argument(token["true_role"]) or is_argument(token["predicted_role"])
    ]
    result = {
        "category": category,
        "instance_index": instance["instance_index"],
        "sentence_id": instance["sentence_id"],
        "sentence": instance["sentence"],
        "predicate": instance["predicate"],
        "arguments": arguments,
        "instance_metrics": instance["metrics"],
    }
    if family_check is not None:
        result["selection_metrics"] = family_summary(instance, family_check)
    else:
        result["selection_metrics"] = {
            "true_arguments": len(true_arguments(instance)),
            "correct_arguments": 0,
        }
    return result


def select_distinct(instances, predicate, count, rank_key):
    candidates = sorted(
        (instance for instance in instances if predicate(instance)),
        key=rank_key,
    )
    selected = []
    used_sentence_ids = set()
    for instance in candidates:
        if instance["sentence_id"] in used_sentence_ids:
            continue
        selected.append(instance)
        used_sentence_ids.add(instance["sentence_id"])
        if len(selected) == count:
            break
    return selected


def success_rank(instance, family_check):
    summary = family_summary(instance, family_check)
    return (
        -summary["true_family_arguments"],
        summary["false_positive_family_arguments"],
        instance["metrics"]["errors"],
        instance["instance_index"],
    )


def failure_rank(instance):
    return (
        -len(true_arguments(instance)),
        instance["metrics"]["token_accuracy"],
        instance["instance_index"],
    )


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Instance analysis not found: {input_path}")
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")

    output_path = Path(
        args.output or input_path.with_name("qualitative_examples.json")
    )
    with input_path.open("r", encoding="utf-8") as input_file:
        instances = json.load(input_file)

    numbered = select_distinct(
        instances,
        lambda instance: all_family_arguments_correct(instance, is_numbered),
        args.count,
        lambda instance: success_rank(instance, is_numbered),
    )
    modifiers = select_distinct(
        instances,
        lambda instance: all_family_arguments_correct(instance, is_modifier),
        args.count,
        lambda instance: success_rank(instance, is_modifier),
    )
    total_failures = select_distinct(
        instances,
        all_arguments_wrong,
        args.count,
        failure_rank,
    )

    output = {
        "criteria": {
            "numbered_successes": (
                "At least one true ARG* (excluding ARGM*) and all such arguments "
                "predicted with the exact role."
            ),
            "modifier_successes": (
                "At least one true ARGM* and all such arguments predicted with "
                "the exact role."
            ),
            "total_failures": (
                "At least one true semantic argument and none predicted with "
                "the exact role."
            ),
            "distinct_sentences": True,
        },
        "numbered_successes": [
            compact_instance(instance, "numbered_success", is_numbered)
            for instance in numbered
        ],
        "modifier_successes": [
            compact_instance(instance, "modifier_success", is_modifier)
            for instance in modifiers
        ],
        "total_failures": [
            compact_instance(instance, "total_failure")
            for instance in total_failures
        ],
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)

    print(
        f"Saved {len(numbered)} numbered successes, "
        f"{len(modifiers)} modifier successes and "
        f"{len(total_failures)} total failures to {output_path}"
    )


if __name__ == "__main__":
    main()
