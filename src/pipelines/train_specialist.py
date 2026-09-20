import json
import os
import sys
import numpy as np
import torch.distributed as dist
os.environ["HF_HOME"] = "/app/.hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback,
    EarlyStoppingCallback,
)

from src.data.srl_data_module import SRLDataModule
from src.training.class_weights import compute_smoothed_class_weights
from src.training.metrics import SRLMetrics
from src.utils.input_reader import define_exp_config

EXPERIMENT_NAME = "srl-portuguese"
EARLY_STOPPING_PATIENCE = 10


class TextLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("TRAINING LOGS\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_path, "a", encoding="utf-8") as f:
                prefix = f"[Epoch {state.epoch:.2f} / Step {state.global_step}] "
                metrics = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
                f.write(prefix + metrics + "\n")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        logits = logits.view(-1, num_classes)
        targets = targets.view(-1)

        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_pt = F.log_softmax(logits, dim=-1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        return loss.mean()


class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_strategy="baseline", class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_strategy = loss_strategy
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        num_labels = logits.size(-1)
        flat_logits = logits.view(-1, num_labels)
        flat_labels = labels.view(-1)

        if self.loss_strategy in {"focal_loss", "weighted_focal_loss"}:
            loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weights, ignore_index=-100)
            loss = loss_fct(flat_logits, flat_labels)
        elif self.loss_strategy == "weighted_loss" and self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
            loss = loss_fct(flat_logits, flat_labels)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(flat_logits, flat_labels)

        return (loss, outputs) if return_outputs else loss


def load_specialist_labels(component):
    labels_folder = "numbered" if component == "numbered" else "modifier"
    labels_path = os.path.join(
        "data", "processed", "labels_and_ids_splitted", labels_folder
    )
    with open(os.path.join(labels_path, "label2id.json"), encoding="utf-8") as file:
        label2id = json.load(file)
    with open(os.path.join(labels_path, "id2label.json"), encoding="utf-8") as file:
        id2label = {int(label_id): label for label_id, label in json.load(file).items()}

    expected_ids = set(range(len(label2id)))
    if set(label2id.values()) != expected_ids or set(id2label) != expected_ids:
        raise ValueError(
            f"Specialist label IDs for '{component}' must be contiguous from 0."
        )
    if any(id2label[label_id] != label for label, label_id in label2id.items()):
        raise ValueError(f"label2id and id2label disagree for '{component}'.")
    if "O" not in label2id or "PRED" not in label2id:
        raise ValueError(f"Specialist vocabulary '{component}' must contain O and PRED.")
    return label2id, id2label


def remap_dataset_labels(dataset, source_id2label, specialist_label2id):
    label_o_id = specialist_label2id["O"]

    def filter_example(example):
        new_labels = []
        for l_id in example["labels"]:
            l_id_int = l_id.item() if hasattr(l_id, "item") else int(l_id)
            if l_id_int == -100:
                new_labels.append(-100)
                continue
            label_name = source_id2label.get(
                l_id_int, source_id2label.get(str(l_id_int))
            )
            new_labels.append(specialist_label2id.get(label_name, label_o_id))

        example["labels"] = new_labels
        return example

    return dataset.map(filter_example, batched=False, load_from_cache_file=False)

from collections import Counter



def main(model_name, num_epochs, batch_size, component, strategy="baseline", seed=42,
         early_stopping_patience=EARLY_STOPPING_PATIENCE):
    assert component in ["numbered", "modifiers"], "component deve ser 'numbered' ou 'modifiers'"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)

    output_path = f"artifacts/{model_name.split('/')[-1]}/{strategy}/seed{seed}"
    run_base_name = f"{model_name.split('/')[-1]}_{strategy}_seed{seed}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log_file_path = f"{output_path}/training_logs_{component}.txt"

    data_module = SRLDataModule(
        raw_dataset_path="data/raw/PBP-classic-complete.conllu",
        model_name=model_name,
        predicate_signal="special_token",
    )
    source_id2label = data_module.id2label
    label2id, id2label = load_specialist_labels(component)

    print(f"Remapping dataset to the {component} specialist vocabulary")
    ds_train = remap_dataset_labels(data_module.datasets["train"], source_id2label, label2id)
    ds_val = remap_dataset_labels(data_module.datasets["validation"], source_id2label, label2id)
    ds_test = remap_dataset_labels(data_module.datasets["test"], source_id2label, label2id)

    class_weights = None
    if strategy == "weighted_focal_loss":
        class_weights, class_counts = compute_smoothed_class_weights(
            ds_train, len(label2id)
        )
        if local_rank == 0:
            print("Smoothed class weights (filtered training split only):")
            for label, label_id in label2id.items():
                print(
                    f"  {label}: count={class_counts[label_id].item()} "
                    f"weight={class_weights[label_id].item():.6f}"
                )

    data_collator = DataCollatorForTokenClassification(tokenizer=data_module.tokenizer.tokenizer, padding=True)
    metrics_calculator = SRLMetrics(id2label=id2label)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model.resize_token_embeddings(len(data_module.tokenizer.tokenizer), mean_resizing=True)

    training_args = TrainingArguments(
        output_dir=f"{output_path}/checkpoints_{component}",
        run_name=f"{run_base_name}_{component}",
        ddp_find_unused_parameters=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        learning_rate=3e-5,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        loss_strategy=strategy,
        class_weights=class_weights,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=data_module.tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics,
        callbacks=[
            TextLoggerCallback(log_file_path),
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
        ]
    )

    print(f"Training {component} model...")
    trainer.train()
    trainer.save_model(f"{output_path}/model_{component}")

    metrics_calculator.eval_mode = True
    val_metrics = trainer.evaluate(ds_val, metric_key_prefix="best_val")
    test_metrics = trainer.evaluate(ds_test, metric_key_prefix="test")

    if local_rank == 0:
        mlflow.start_run(run_name=f"{run_base_name}_{component}")
        mlflow.set_tags({"component": component, "strategy": strategy})
        mlflow.log_params({
            "model_name": model_name, "strategy": strategy, "seed": seed,
            "batch_size": batch_size, "num_epochs_ceiling": num_epochs,
            "early_stopping_patience": early_stopping_patience,
        })
        final_metrics = {**val_metrics, **test_metrics}
        mlflow.log_metrics({
            key: value
            for key, value in final_metrics.items()
            if isinstance(value, (int, float))
        })

        metrics_path = f"{output_path}/final_metrics_{component}.json"
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(final_metrics, metrics_file, indent=4, ensure_ascii=False)

        mlflow.log_artifact(log_file_path)
        mlflow.log_artifact(metrics_path)
        mlflow.end_run()

        print(f"Saved {component} model at {output_path}/model_{component}")
        print(f"Test metrics ({component}): {test_metrics}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    model_name, model_size, num_epochs, batch_size, strategy, seed = define_exp_config()

    try:
        component = sys.argv[7].replace("--", "")
    except IndexError:
        raise ValueError(
            "Missing --component argument (numbered | modifiers) as the 7th argument."
        )

    if component not in {"numbered", "modifiers"}:
        raise ValueError(f"Invalid component '{component}'. Valid: numbered, modifiers.")
    
    cfg_path = f"src/configs/{model_name}.json"
    cfg = json.load(open(cfg_path))
    cfg = cfg[model_size]

    main(
        cfg["model_name"],
        num_epochs=num_epochs,
        batch_size=batch_size,
        component=component,
        strategy=strategy,
        seed=seed,
    )