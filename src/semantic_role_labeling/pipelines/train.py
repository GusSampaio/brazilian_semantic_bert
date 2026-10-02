import json
import os
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

MLFLOW_EXPERIMENT_NAME = "srl-portuguese"
EARLY_STOPPING_PATIENCE=10
MODEL_PIP_REQUIREMENTS = [
    f"{package}=={version(package)}"
    for package in ("mlflow", "torch", "transformers", "tokenizers", "safetensors")
]

def main(model_name, num_epochs, batch_size, strategy="baseline", seed=42, early_stopping_patience=EARLY_STOPPING_PATIENCE):
    if strategy not in {"baseline", "focal_loss"}:
        raise ValueError(
            "Single-model training strategy must be 'baseline' or 'focal_loss'."
        )

    model_slug = model_name.split("/")[-1]
    output_path = f"artifacts/{model_slug}/{strategy}/seed{seed}"
    log_file_path = f"{output_path}/training_logs.txt"
    run_name = f"{model_slug}_{strategy}_seed{seed}"

    data_module = SRLDataModule(
        raw_dataset_path="data/raw/PBP-classic-complete.conllu",
        model_name=model_name,
        predicate_signal="special_token",
    )
    
    hf_datasets = data_module.datasets
    train_dataset, validation_dataset, test_dataset = hf_datasets["train"], hf_datasets["validation"], hf_datasets["test"]

    model = AutoModelForTokenClassification.from_pretrained(
        data_module.model_name,
        num_labels=len(data_module.label2id),
        id2label=data_module.id2label,
        label2id=data_module.label2id
    )

    model.resize_token_embeddings(len(data_module.tokenizer.tokenizer),
                                  mean_resizing=True)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=data_module.tokenizer.tokenizer,
        padding=True # dynamic padding 
    )

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False, # Avoids errors with DDP when using multiple GPUs
        run_name=run_name,

        output_dir=f"{output_path}/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        fp16=torch.cuda.is_available(), # Use of mixed precision if using GPU
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

    metrics_calculator = SRLMetrics(id2label=data_module.id2label)
    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    is_main_process = int(os.environ.get("RANK","0")) == 0

    if is_main_process:
        callbacks.append(TextLoggerCallback(log_file_path))
        callbacks.append(MLflowCallback())

    trainer = SRLTrainer(
        model=model,
        args=training_args,
        loss_strategy=strategy,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=data_module.tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics, 
        callbacks= callbacks
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    if is_main_process:
        print(f"Starting run: {run_name}")
        mlflow.start_run(run_name=run_name)


    trainer.train()

    final_model_dir = f"{output_path}/final_model"
    trainer.save_model(final_model_dir)
    print(f"End of training! Saved at {final_model_dir}")

    trainer.pop_callback(EarlyStoppingCallback)

    metrics_calculator.eval_mode = True
    val_metrics = trainer.evaluate(validation_dataset, metric_key_prefix="best_val")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    
    if is_main_process:
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_params({
                "strategy": strategy,
                "num_epochs_ceiling": num_epochs,
                "early_stopping_patience": early_stopping_patience,
            })

            metrics_path = f"{output_path}/final_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({**val_metrics, **test_metrics}, f, indent=4)


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

            mlflow.end_run()

        print(f"Métricas finais (teste): {test_metrics}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    model_name, model_size, num_epochs, batch_size, strategy, seed = define_exp_config()
    cfg_path = f"src/semantic_role_labeling/configs/{model_name}.json"
    cfg = json.load(open(cfg_path))
    cfg = cfg[model_size]

    main(
        cfg["model_name"],
        num_epochs=num_epochs,
        batch_size=batch_size,
        strategy=strategy,
        seed=seed,
    )