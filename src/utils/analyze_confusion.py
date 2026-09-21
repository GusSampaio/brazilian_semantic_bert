import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)
from transformers import (AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments)

from src.data.conllu_parser import PBP_parser
from src.data.srl_data_module import SRLDataModule

DEFAULT_MODEL_PATH = "artifacts/xlm-roberta-large/baseline/seed120/final_model"
DEFAULT_BASE_MODEL = "xlm-roberta-large"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SRL model and generate confusion matrices."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()

def label_for_id(id2label, label_id):
    return id2label.get(int(label_id), id2label.get(str(int(label_id)), "O"))

def validate_label_mapping(model, data_module):
    model_mapping = {int(label_id): label for label_id, label in model.config.id2label.items()}
    dataset_mapping = {int(label_id): label for label_id, label in data_module.id2label.items()}

    if model_mapping != dataset_mapping:
        raise ValueError(
            "Model and reconstructed dataset use different id2label mappings. "
            "The resulting confusion matrix would be invalid."
        )

def save_matrix_csv(path, matrix, row_labels, column_labels):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["true\\pred", *column_labels])
        for label, row in zip(row_labels, matrix):
            writer.writerow([label, *row.tolist()])

def save_matrix_plot(path, matrix, row_labels, column_labels, title, annotate=False):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    width = max(7, 0.55 * len(column_labels))
    height = max(5, 0.55 * len(row_labels))
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    figure.colorbar(image, ax=axis, label="Proportion within true class")
    axis.set(
        xticks=np.arange(len(column_labels)),
        yticks=np.arange(len(row_labels)),
        xticklabels=column_labels,
        yticklabels=row_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if annotate:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="white" if normalized[row_index, column_index] > 0.5 else "black",
                )

    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def softmax(logits):
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)

def reconstruct_test_instances(data_module):
    instances = PBP_parser(data_module.raw_dataset_path)
    processed_instances = data_module.builder.build(instances)
    _, _, test_instances = data_module.splitter.split(processed_instances)
    return test_instances

def safe_mean(values):
    return float(np.mean(values)) if values else None

def build_instance_analysis(instance_index, instance, dataset_example, logits, predicted_ids, true_ids, tokenizer, id2label, argument_labels):
    sequence_length = len(dataset_example["input_ids"])
    input_ids = dataset_example["input_ids"].tolist()
    example_true_ids = true_ids[:sequence_length]
    example_predicted_ids = predicted_ids[:sequence_length]
    probabilities = softmax(logits[:sequence_length])

    encoding = tokenizer(
        instance["model_tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=sequence_length,
        spaces_between_special_tokens=True,
    )
    if encoding["input_ids"] != input_ids:
        raise ValueError(
            f"Token alignment mismatch at test instance {instance_index}."
        )

    word_ids = encoding.word_ids()
    model_subtokens = tokenizer.convert_ids_to_tokens(input_ids)
    token_analysis = []
    true_labels = []
    predicted_labels = []
    correct_confidences = []
    error_confidences = []

    for position, (token_id, subtoken, word_id, true_id, predicted_id) in enumerate(
        zip(
            input_ids,
            model_subtokens,
            word_ids,
            example_true_ids,
            example_predicted_ids,
        )
    ):
        ignored = int(true_id) == -100
        true_label = None if ignored else label_for_id(id2label, true_id)
        predicted_label = None if ignored else label_for_id(id2label, predicted_id)
        confidence = None if ignored else float(probabilities[position, predicted_id])
        true_label_probability = (
            None if ignored else float(probabilities[position, true_id])
        )
        correct = None if ignored else true_label == predicted_label

        if not ignored:
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            if correct:
                correct_confidences.append(confidence)
            else:
                error_confidences.append(confidence)

        token_analysis.append(
            {
                "position": position,
                "token_id": token_id,
                "subtoken": subtoken,
                "source_token_index": word_id,
                "source_token": (
                    instance["model_tokens"][word_id] if word_id is not None else None
                ),
                "ignored_in_evaluation": ignored,
                "true_role": true_label,
                "predicted_role": predicted_label,
                "confidence": confidence,
                "true_role_probability": true_label_probability,
                "correct": correct,
            }
        )

    true_labels_array = np.asarray(true_labels)
    predicted_labels_array = np.asarray(predicted_labels)
    true_is_argument = np.isin(true_labels_array, argument_labels)
    predicted_is_argument = np.isin(predicted_labels_array, argument_labels)
    identification_precision, identification_recall, identification_f1, _ = (
        precision_recall_fscore_support(
            true_is_argument,
            predicted_is_argument,
            average="binary",
            zero_division=0,
        )
    )
    argument_mask = true_is_argument
    argument_true = true_labels_array[argument_mask]
    argument_predicted = predicted_labels_array[argument_mask]
    roles_in_instance = sorted(set(argument_true.tolist()))

    errors = [
        {
            "position": token["position"],
            "subtoken": token["subtoken"],
            "source_token": token["source_token"],
            "true_role": token["true_role"],
            "predicted_role": token["predicted_role"],
            "confidence": token["confidence"],
        }
        for token in token_analysis
        if token["correct"] is False
    ]
    correct_count = sum(true == predicted for true, predicted in zip(true_labels, predicted_labels))

    return {
        "instance_index": instance_index,
        "sentence_id": instance["sentence_id"],
        "sentence": instance["text"],
        "predicate": {
            "token": instance["predicate_token"],
            "token_index": instance["predicate_index"],
            "lemma": instance["predicate_lemma"],
            "frame": instance["predicate_frame"],
        },
        "original_tokens": instance["tokens"],
        "original_true_roles": instance["labels"],
        "model_input_tokens": instance["model_tokens"],
        "model_subtokens": token_analysis,
        "errors": errors,
        "metrics": {
            "evaluated_tokens": len(true_labels),
            "correct_tokens": correct_count,
            "errors": len(errors),
            "token_accuracy": correct_count / len(true_labels) if true_labels else None,
            "exact_match": not errors,
            "identification_precision": float(identification_precision),
            "identification_recall": float(identification_recall),
            "identification_f1": float(identification_f1),
            "true_arguments": int(true_is_argument.sum()),
            "predicted_arguments": int(predicted_is_argument.sum()),
            "role_accuracy_on_true_arguments": (
                float(accuracy_score(argument_true, argument_predicted))
                if len(argument_true)
                else None
            ),
            "role_macro_f1_on_true_arguments": (
                float(
                    f1_score(
                        argument_true,
                        argument_predicted,
                        labels=roles_in_instance,
                        average="macro",
                        zero_division=0,
                    )
                )
                if roles_in_instance
                else None
            ),
            "mean_confidence_correct": safe_mean(correct_confidences),
            "mean_confidence_errors": safe_mean(error_confidences),
        },
    }

def save_instance_analysis(path, test_instances, test_dataset, predictions, predicted_ids, true_ids, tokenizer, id2label, argument_labels):
    if len(test_instances) != len(test_dataset):
        raise ValueError(
            "Reconstructed instances and tokenized test dataset have different sizes."
        )

    analyses = []
    for instance_index, (instance, dataset_example) in enumerate(
        zip(test_instances, test_dataset)
    ):
        analyses.append(
            build_instance_analysis(
                instance_index=instance_index,
                instance=instance,
                dataset_example=dataset_example,
                logits=predictions[instance_index],
                predicted_ids=predicted_ids[instance_index],
                true_ids=true_ids[instance_index],
                tokenizer=tokenizer,
                id2label=id2label,
                argument_labels=argument_labels,
            )
        )

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(analyses, output_file, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    output_dir = Path(args.output_dir or model_path.parent / "confusion_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reconstructing test dataset...")
    data_module = SRLDataModule(
        raw_dataset_path="data/raw/PBP-classic-complete.conllu",
        model_name=args.base_model,
        predicate_signal="special_token",
    )
    test_dataset = data_module.datasets["test"]
    test_instances = reconstruct_test_instances(data_module)

    print(f"Loading model from {model_path}...")
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    validate_label_mapping(model, data_module)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "tmp"),
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
        ),
        data_collator=DataCollatorForTokenClassification(
            tokenizer=data_module.tokenizer.tokenizer,
            padding=True,
        ),
        processing_class=data_module.tokenizer.tokenizer,
    )

    print(f"Running inference on {len(test_dataset)} test examples...")
    prediction_output = trainer.predict(test_dataset)
    predicted_ids = np.argmax(prediction_output.predictions, axis=-1)
    true_ids = prediction_output.label_ids
    valid_mask = true_ids != -100
    flat_true_ids = true_ids[valid_mask]
    flat_predicted_ids = predicted_ids[valid_mask]

    id2label = data_module.id2label
    ordered_labels = [
        label_for_id(id2label, label_id) for label_id in sorted(map(int, id2label))
    ]
    true_labels = np.array([label_for_id(id2label, value) for value in flat_true_ids])
    predicted_labels = np.array(
        [label_for_id(id2label, value) for value in flat_predicted_ids]
    )
    argument_labels = [
        label for label in ordered_labels if label not in {"O", "PRED"}
    ]

    print("Saving instance-level analysis...")
    save_instance_analysis(
        path=output_dir / "instance_analysis.json",
        test_instances=test_instances,
        test_dataset=test_dataset,
        predictions=prediction_output.predictions,
        predicted_ids=predicted_ids,
        true_ids=true_ids,
        tokenizer=data_module.tokenizer.tokenizer,
        id2label=id2label,
        argument_labels=argument_labels,
    )

    full_matrix = confusion_matrix(
        true_labels, predicted_labels, labels=ordered_labels
    )
    save_matrix_csv(
        output_dir / "confusion_full.csv",
        full_matrix,
        ordered_labels,
        ordered_labels,
    )
    save_matrix_plot(
        output_dir / "confusion_full.png",
        full_matrix,
        ordered_labels,
        ordered_labels,
        "SRL confusion matrix - all labels",
    )

    true_is_argument = np.isin(true_labels, argument_labels)
    predicted_is_argument = np.isin(predicted_labels, argument_labels)
    identification_matrix = confusion_matrix(
        true_is_argument,
        predicted_is_argument,
        labels=[False, True],
    )
    identification_labels = ["not_argument", "argument"]
    save_matrix_csv(
        output_dir / "confusion_identification.csv",
        identification_matrix,
        identification_labels,
        identification_labels,
    )
    save_matrix_plot(
        output_dir / "confusion_identification.png",
        identification_matrix,
        identification_labels,
        identification_labels,
        "Argument identification",
        annotate=True,
    )

    identification_precision, identification_recall, identification_f1, _ = (
        precision_recall_fscore_support(
            true_is_argument,
            predicted_is_argument,
            average="binary",
            zero_division=0,
        )
    )
    argument_true_labels = true_labels[true_is_argument]
    argument_predicted_labels = predicted_labels[true_is_argument]
    report = {
        "model_path": str(model_path),
        "test_examples": len(test_dataset),
        "evaluated_tokens": int(valid_mask.sum()),
        "identification": {
            "precision": float(identification_precision),
            "recall": float(identification_recall),
            "f1": float(identification_f1),
        },
        "classification_on_true_arguments": {
            "accuracy": float(
                accuracy_score(argument_true_labels, argument_predicted_labels)
            ),
            "macro_f1": float(
                f1_score(
                    argument_true_labels,
                    argument_predicted_labels,
                    labels=argument_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)

    with (output_dir / "token_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["true_label", "predicted_label"])
        writer.writerows(zip(true_labels.tolist(), predicted_labels.tolist()))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Analysis saved to {output_dir}")

if __name__ == "__main__":
    main()
