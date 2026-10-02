import json
import os
import sys
from importlib.metadata import version

import torch.distributed as dist
os.environ["HF_HOME"] = "/app/.hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import torch
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from transformers.integrations import MLflowCallback

from src.semantic_role_labeling.data.srl_data_module import SRLDataModule
from src.semantic_role_labeling.training.callbacks import TextLoggerCallback
from src.semantic_role_labeling.training.metrics import SRLMetrics
from src.semantic_role_labeling.training.trainer import SRLTrainer
from src.semantic_role_labeling.utils.input_reader import define_exp_config

EXPERIMENT_NAME = "srl-portuguese"
EARLY_STOPPING_PATIENCE = 10
SPECIALIST_STRATEGY = "specialists_ensemble"
SPECIALIST_LOSS_STRATEGY = "baseline"
MODEL_PIP_REQUIREMENTS = [
    f"{package}=={version(package)}"
    for package in ("mlflow", "torch", "transformers", "tokenizers", "safetensors")
]


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

def main(model_name, num_epochs, batch_size, component, strategy=SPECIALIST_STRATEGY, seed=42,
         early_stopping_patience=EARLY_STOPPING_PATIENCE):
    assert component in ["numbered", "modifiers"], "component deve ser 'numbered' ou 'modifiers'"
    if strategy != SPECIALIST_STRATEGY:
        raise ValueError(
            f"Specialist models must use strategy '{SPECIALIST_STRATEGY}' so their "
            "artifacts remain separate from single-model experiments."
        )

    output_path = f"artifacts/{model_name.split('/')[-1]}/{strategy}/seed{seed}"
    run_base_name = f"{model_name.split('/')[-1]}_{strategy}_seed{seed}"
    run_name = f"{run_base_name}_{component}"
    is_main_process = int(os.environ.get("RANK", "0")) == 0
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

    data_collator = DataCollatorForTokenClassification(tokenizer=data_module.tokenizer.tokenizer, padding=True)
    metrics_calculator = SRLMetrics(id2label=id2label)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model.resize_token_embeddings(len(data_module.tokenizer.tokenizer), mean_resizing=True)

    training_args = TrainingArguments(
        output_dir=f"{output_path}/checkpoints_{component}",
        run_name=run_name,
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

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
    ]
    if is_main_process:
        callbacks.append(TextLoggerCallback(log_file_path))
        callbacks.append(MLflowCallback())

    trainer = SRLTrainer(
        model=model,
        args=training_args,
        loss_strategy=SPECIALIST_LOSS_STRATEGY,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=data_module.tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics,
        callbacks=callbacks,
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)
    if is_main_process:
        print(f"Starting run: {run_name}")
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags({"component": component, "strategy": strategy})

    try:
        print(f"Training {component} model...")
        trainer.train()

        final_model_dir = f"{output_path}/model_{component}"
        trainer.save_model(final_model_dir)
        print(f"Saved {component} model at {final_model_dir}")

        trainer.pop_callback(EarlyStoppingCallback)
        metrics_calculator.eval_mode = True
        val_metrics = trainer.evaluate(ds_val, metric_key_prefix="best_val")
        test_metrics = trainer.evaluate(ds_test, metric_key_prefix="test")

        if is_main_process and mlflow.active_run():
            final_metrics = {**val_metrics, **test_metrics}
            metrics_path = f"{output_path}/final_metrics_{component}.json"
            with open(metrics_path, "w", encoding="utf-8") as metrics_file:
                json.dump(final_metrics, metrics_file, indent=4, ensure_ascii=False)

            mlflow.log_artifact(log_file_path)
            mlflow.log_artifact(metrics_path)
            mlflow.transformers.log_model(
                transformers_model={
                    "model": trainer.model,
                    "tokenizer": data_module.tokenizer.tokenizer,
                },
                name="model",
                pip_requirements=MODEL_PIP_REQUIREMENTS,
            )
            print(f"Test metrics ({component}): {test_metrics}")
    finally:
        if is_main_process and mlflow.active_run():
            mlflow.end_run()
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
    
    cfg_path = f"src/semantic_role_labeling/configs/{model_name}.json"
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